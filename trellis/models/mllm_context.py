import os
from typing import List, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import torch.nn as nn
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from trellis.modules.utils import zero_module


class MLLMContextModel(nn.Module):
    """
    Spatial3D MLLM Context Model.
    """

    QUERY_TYPES = ["ss", "slat"]

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B",
        num_queries: int = 32,
        query_types: List[str] = QUERY_TYPES,
        cond_channels: int = 1024,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.cond_channels = cond_channels

        # process mllm model
        self.mllm_backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        num_embeddings = self.mllm_backbone.get_input_embeddings().num_embeddings
        self.num_embeddings = num_embeddings

        self.mllm_backbone.resize_token_embeddings(
            num_embeddings + (num_queries + 2) * len(self.QUERY_TYPES),
            mean_resizing=False,
        )

        def freeze_hook(grad):
            grad[: self.num_embeddings].zero_()
            return grad

        self.mllm_backbone.language_model.embed_tokens.weight.register_hook(freeze_hook)
        self.mllm_backbone.lm_head = nn.Identity()
        self.mllm_backbone.eval()

        # process processor
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        vocab_size = len(self.processor.tokenizer)

        # pad placeholder tokens in mllm
        self.processor.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    f"<pad_token_{i}>" for i in range(num_embeddings - vocab_size)
                ]
            }
        )

        # add special tokens for queries
        special_tokens = []
        for qtype in self.QUERY_TYPES:
            special_tokens.append(f"<|begin_of_{qtype}|>")
            special_tokens.append(f"<|end_of_{qtype}|>")
            special_tokens.extend(f"<|{qtype}_{i}|>" for i in range(num_queries))

        self.processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )

        self.boi_token_ids = {
            qtype: self.processor.tokenizer.convert_tokens_to_ids(
                f"<|begin_of_{qtype}|>"
            )
            for qtype in self.QUERY_TYPES
        }

        self.eoi_token_ids = {
            qtype: self.processor.tokenizer.convert_tokens_to_ids(f"<|end_of_{qtype}|>")
            for qtype in self.QUERY_TYPES
        }

        self.connector = nn.ModuleDict()

        for qtype in query_types:
            self.connector[qtype] = nn.Sequential(
                nn.Linear(
                    self.mllm_backbone.config.hidden_size,
                    self.cond_channels,
                ),
                nn.GELU(approximate="tanh"),
                zero_module(nn.Linear(self.cond_channels, self.cond_channels)),
            )

        # freeze mllm except for newly added tokens
        for key, param in self.mllm_backbone.named_parameters():
            if key != "model.language_model.embed_tokens.weight":
                param.requires_grad_(False)

        self.instruction = "Generate a 3D scene."
        self.responses = {
            qtype: f"<|begin_of_{qtype}|>"
            + "".join([f"<|{qtype}_{i}|>" for i in range(num_queries)])
            + f"<|end_of_{qtype}|>"
            for qtype in self.QUERY_TYPES
        }

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.mllm_backbone.parameters()).device

    @torch.no_grad()
    def tokenize(
        self,
        captions: Optional[List[str]] = None,
        scripts: Optional[List[str]] = None,
        images: Optional[List[Image.Image]] = None,
        query_type: str = "ss",
    ):
        if not isinstance(captions, List):
            captions = [captions]

        if not isinstance(scripts, List):
            scripts = [scripts]

        if not isinstance(images, List):
            images = [images]

        messages = []
        for caption, script, image in zip(captions, scripts, images):

            user_contents = [{"type": "text", "text": self.instruction}]

            conditions = {"Caption": caption, "Script": script, "Image": image}

            for key, content in conditions.items():

                if content is None:
                    continue

                user_contents.append({"type": "text", "text": f"{key}: {content}\n"})

                if key == "Image":
                    user_contents.append({"type": "image", "image": content})

            messages.append(
                [
                    {
                        "role": "user",
                        "content": (
                            user_contents
                            if len(user_contents) > 1
                            else user_contents[0]["text"]
                        ),
                    },
                    {"role": "assistant", "content": self.responses[query_type]},
                ]
            )

        texts = self.processor.apply_chat_template(
            messages,
            tokenize=False,
        )

        image_inputs, _ = process_vision_info(messages)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        return inputs

    def forward(
        self,
        captions: Optional[List[str]] = None,
        scripts: Optional[List[str]] = None,
        images: Optional[List[Image.Image]] = None,
        query_type: str = "ss",
    ) -> torch.Tensor:

        inputs = self.tokenize(
            captions=captions,
            scripts=scripts,
            images=images,
            query_type=query_type,
        )

        query_embeds = self.mllm_backbone(**inputs).logits
        query_embeds = query_embeds.type(torch.float32)

        input_ids = inputs["input_ids"]

        boi_pos = torch.where(input_ids == self.boi_token_ids[query_type])[1]
        eoi_pos = torch.where(input_ids == self.eoi_token_ids[query_type])[1]

        # Create mask for selecting tokens between BOI and EOI
        batch_size, seq_len = input_ids.shape
        indices = torch.arange(seq_len, device=input_ids.device)[None, :].expand(
            batch_size, -1
        )
        mask = (indices > boi_pos[:, None]) & (indices < eoi_pos[:, None])

        query_embeds = query_embeds[mask].view(batch_size, -1, query_embeds.size(-1))
        query_embeds = self.connector[query_type](query_embeds)

        return query_embeds
