import os
import copy
from typing import Union, List
from collections import OrderedDict

import torch
from loguru import logger

from ..vram.initialization import skip_model_initialization
from ..vram.disk_map import DiskMap
from ..vram.layers import enable_vram_management
from .file import load_state_dict

def _load_state_dict_into_model(model_to_load, state_dict: OrderedDict) -> List[str]:
    # Convert old format to new format if needed from a PyTorch state_dict
    # copy state_dict so _load_from_state_dict can modify it
    state_dict = state_dict.copy()
    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: torch.nn.Module, prefix: str = ""):
        # Workaround for meta device
        for name, param in module._parameters.items():
            if param is not None and param.device.type == "meta" and prefix + name in state_dict:
                if state_dict[prefix + name].shape != param.shape:
                    continue
                module.register_parameter(name, torch.nn.Parameter(state_dict[prefix + name], requires_grad=param.requires_grad))
        for name, buffer in module._buffers.items():
            if buffer is not None and buffer.device.type == "meta" and prefix + name in state_dict:
                if state_dict[prefix + name].shape != buffer.shape:
                    continue
                module.register_buffer(name, state_dict[prefix + name])

        args = (state_dict, prefix, {}, True, [], [], error_msgs)
        module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model_to_load)

    return error_msgs

def _load_pretrained_model(
    model: torch.nn.Module,
    state_dict: OrderedDict,
    pretrained_model_name_or_path: Union[str, os.PathLike],
    ignore_mismatched_sizes: bool = False,
):
    # Retrieve missing & unexpected_keys
    model_state_dict = model.state_dict()
    loaded_keys = list(state_dict.keys())

    expected_keys = list(model_state_dict.keys())

    original_loaded_keys = loaded_keys

    missing_keys = list(set(expected_keys) - set(loaded_keys))
    unexpected_keys = list(set(loaded_keys) - set(expected_keys))

    # Make sure we are able to load base models as well as derived models (with heads)
    model_to_load = model

    def _find_mismatched_keys(
        state_dict,
        model_state_dict,
        loaded_keys,
        ignore_mismatched_sizes,
    ):
        mismatched_keys = []
        if ignore_mismatched_sizes:
            for checkpoint_key in loaded_keys:
                model_key = checkpoint_key

                if (
                    model_key in model_state_dict
                    and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                ):
                    mismatched_keys.append(
                        (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                    )
                    del state_dict[checkpoint_key]
        return mismatched_keys

    if state_dict is not None:
        # Whole checkpoint
        mismatched_keys = _find_mismatched_keys(
            state_dict,
            model_state_dict,
            original_loaded_keys,
            ignore_mismatched_sizes,
        )
    
    error_msgs = _load_state_dict_into_model(model_to_load, state_dict)

    if len(error_msgs) > 0:
        error_msg = "\n\t".join(error_msgs)
        if "size mismatch" in error_msg:
            error_msg += (
                "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
            )
        raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

    if len(unexpected_keys) > 0:
        logger.warning(
            f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
            f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
            f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task"
            " or with another architecture (e.g. initializing a BertForSequenceClassification model from a"
            " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
            f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly"
            " identical (initializing a BertForSequenceClassification model from a"
            " BertForSequenceClassification model)."
        )
    else:
        logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
            " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
        )
    elif len(mismatched_keys) == 0:
        logger.info(
            f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the"
            f" checkpoint was trained on, you can already use {model.__class__.__name__} for predictions"
            " without further training."
        )
    if len(mismatched_keys) > 0:
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        logger.warning(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
            f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be"
            " able to use it for predictions and inference."
        )

    return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs


def load_model(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, use_disk_map=False, module_map=None, vram_config=None, vram_limit=None, zero_init_conv_in=True):
    config = {} if config is None else config
    # Why do we use `skip_model_initialization`?
    # It skips the random initialization of model parameters,
    # thereby speeding up model loading and avoiding excessive memory usage.
    with skip_model_initialization():
        model = model_class(**config)
    # What is `module_map`?
    # This is a module mapping table for VRAM management.
    if module_map is not None:
        devices = [vram_config["offload_device"], vram_config["onload_device"], vram_config["preparing_device"], vram_config["computation_device"]]
        device = [d for d in devices if d != "disk"][0]
        dtypes = [vram_config["offload_dtype"], vram_config["onload_dtype"], vram_config["preparing_dtype"], vram_config["computation_dtype"]]
        dtype = [d for d in dtypes if d != "disk"][0]
        if vram_config["offload_device"] != "disk":
            state_dict = DiskMap(path, device, torch_dtype=dtype)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            else:
                state_dict = {i: state_dict[i] for i in state_dict}
            model.load_state_dict(state_dict, assign=True)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=None, vram_limit=vram_limit)
            logger.info(f"Model loaded with VRAM_config in offload_device {vram_config['offload_device']} from {path}")
        else:
            disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=vram_limit)
            logger.info(f"Model loaded with VRAM_config in offload_device disk from {path}")
    else:
        # Why do we use `DiskMap`?
        # Sometimes a model file contains multiple models,
        # and DiskMap can load only the parameters of a single model,
        # avoiding the need to load all parameters in the file.
        if use_disk_map:
            state_dict = DiskMap(path, device, torch_dtype=torch_dtype)
        else:
            state_dict = load_state_dict(path, torch_dtype, device)
        # Why do we use `state_dict_converter`?
        # Some models are saved in complex formats,
        # and we need to convert the state dict into the appropriate format.
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        else:
            state_dict = {i: state_dict[i] for i in state_dict}
        
        state_dict_original = copy.deepcopy(state_dict)
        # cope with the situation where the state_dict keys are mismatched with model keys
        model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = _load_pretrained_model(model=model,
                                                                                                state_dict=state_dict,
                                                                                                pretrained_model_name_or_path=path,
                                                                                                ignore_mismatched_sizes=True)
        if len(mismatched_keys) > 0:
            for key, ckpt_shape, model_shape in mismatched_keys:
                # patch_embedding.weight: found shape torch.Size([1536, 64, 1, 2, 2]) in the checkpoint and torch.Size([1536, 64, 1, 2, 2]) in the model_state
                if key == "patch_embedding.weight" and ckpt_shape[0] == model_shape[0]:
                    # Zero-initialize the extra patch embedding weights
                    old_weight = state_dict_original[key]
                    latent_channels = old_weight.shape[1]  # [1536, 48, 1, 2, 2]
                    avg_old_weight = old_weight.mean(dim=1, keepdim=True)  # [1536, 1, 1, 2, 2]
                    if model.patch_embedding.weight.data.shape[1] != latent_channels:
                        # Initialize from the original weights
                        model.patch_embedding.weight.data[:, :latent_channels] = old_weight
                        # 从48个通道中随机采样16个
                        sampled_indices = torch.randint(0, 48, (model_shape[1] - latent_channels,))
                        sampled_weights = old_weight[:, sampled_indices]  # [1536, 16, 1, 2, 2]
                        # 添加2%的小噪声打破完全重复
                        noise = torch.randn_like(sampled_weights) * old_weight.std() * 0.02
                        sampled_weights = sampled_weights + noise
                        model.patch_embedding.weight.data[:, latent_channels:] = sampled_weights
                        # if zero_init_conv_in:
                        #     model.patch_embedding.weight.data[:, latent_channels:] = 0
                        logger.info(f"{model.__class__.__name__} Zero-initialized patch_embedding.weight from shape {old_weight.shape} to {model.patch_embedding.weight.data.shape}")
                elif key == "head.head.weight" or key == "head.head.bias":
                    # head.head.bias: found shape torch.Size([64]) in the checkpoint and torch.Size([128]) in the model
                    # Initialize from the original weights, copy the weights
                    old_weight = state_dict_original[key]
                    out_channels = old_weight.shape[0]  # [64]
                    if key.endswith("weight"):
                        model.head.head.weight.data[:out_channels] = old_weight
                        model.head.head.weight.data[out_channels:] = old_weight
                    elif key.endswith("bias"):
                        model.head.head.bias.data[:out_channels] = old_weight
                        model.head.head.bias.data[out_channels:] = old_weight
                    logger.info(f"{model.__class__.__name__} initialized {key} from shape {ckpt_shape} to {model_shape}")
                    
        
        # Materialize missing parameters that are still on meta device
        def materialize_meta_tensors(module):
            for name, param in module._parameters.items():
                if param is not None and param.device.type == "meta":
                    module.register_parameter(name, torch.nn.Parameter(torch.empty_like(param, device=device, dtype=torch_dtype), requires_grad=param.requires_grad))
            for name, buffer in module._buffers.items():
                if buffer is not None and buffer.device.type == "meta":
                    module.register_buffer(name, torch.empty_like(buffer, device=device, dtype=torch_dtype))
            for child in module.children():
                materialize_meta_tensors(child)
        materialize_meta_tensors(model)
        
        # model.load_state_dict(state_dict, assign=True)
        # Why do we call `to()`?
        # Because some models override the behavior of `to()`,
        # especially those from libraries like Transformers.
        model = model.to(dtype=torch_dtype, device=device)
    if hasattr(model, "eval"):
        model = model.eval()
    return model


def load_model_with_disk_offload(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, module_map=None):
    if isinstance(path, str):
        path = [path]
    config = {} if config is None else config
    with skip_model_initialization():
        model = model_class(**config)
    if hasattr(model, "eval"):
        model = model.eval()
    disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": device,
        "computation_dtype": torch_dtype,
        "computation_device": device,
    }
    enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=80)
    return model
