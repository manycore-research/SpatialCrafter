import importlib

__attributes = {
    "BasicTrainer": "basic",
    "SparseStructureVaeTrainer": "vae.sparse_structure_vae",
    "SLatVaeGaussianTrainer": "vae.structured_latent_vae_gaussian",
    "SFeatVaeGaussianTrainer": "vae.structured_feat_vae_gaussian",
    "SLatVaeRadianceFieldDecoderTrainer": "vae.structured_latent_vae_rf_dec",
    "SLatVaeMeshDecoderTrainer": "vae.structured_latent_vae_mesh_dec",
    "FlowMatchingTrainer": "flow_matching.flow_matching",
    "FlowMatchingCFGTrainer": "flow_matching.flow_matching",
    "MLLMConditionedFlowMatchingCFGTrainer": "flow_matching.flow_matching",
    "ImageConditionedFlowMatchingCFGTrainer": "flow_matching.flow_matching",
    "SparseFlowMatchingTrainer": "flow_matching.sparse_flow_matching",
    "SparseFlowMatchingCFGTrainer": "flow_matching.sparse_flow_matching",
    "MLLMConditionedSparseFlowMatchingCFGTrainer": "flow_matching.sparse_flow_matching",
    "ImageConditionedSparseFlowMatchingCFGTrainer": "flow_matching.sparse_flow_matching_ori",
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == "__main__":
    from trellis.trainers.basic import BasicTrainer
    from trellis.trainers.flow_matching.flow_matching import (
        FlowMatchingCFGTrainer,
        FlowMatchingTrainer,
        MLLMConditionedFlowMatchingCFGTrainer,
        ImageConditionedFlowMatchingCFGTrainer,
    )
    from trellis.trainers.flow_matching.sparse_flow_matching import (
        SparseFlowMatchingCFGTrainer,
        SparseFlowMatchingTrainer,
        MLLMConditionedSparseFlowMatchingCFGTrainer,
    )
    from trellis.trainers.flow_matching.sparse_flow_matching_ori import (
        ImageConditionedSparseFlowMatchingCFGTrainer,
    )
    from trellis.trainers.vae.sparse_structure_vae import SparseStructureVaeTrainer
    from trellis.trainers.vae.structured_latent_vae_gaussian import (
        SLatVaeGaussianTrainer,
    )
    from trellis.trainers.vae.structured_feat_vae_gaussian import (
        SFeatVaeGaussianTrainer,
    )
    from trellis.trainers.vae.structured_latent_vae_mesh_dec import (
        SLatVaeMeshDecoderTrainer,
    )
    from trellis.trainers.vae.structured_latent_vae_rf_dec import (
        SLatVaeRadianceFieldDecoderTrainer,
    )
