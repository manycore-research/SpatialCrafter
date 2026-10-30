# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Data-Engine public release.
- Training code release.
- Code refactor to split the inherited `diffsynth` dependency tree from the
  project-specific code, and to unify the project's version and license metadata.

## [1.0.0] - 2026-10-27

Initial public release, accompanying the SIGGRAPH Asia 2026 paper
*SpatialCrafter: Single Image World Modeling with Generative 3D Proxies*.

### Added

- **Inference pipelines**
  - `scripts/infer_rgbd_video.py`: single image to RGB-D video generation.
  - `scripts/infer_rgb_scm_video.py`: single image to RGB-SCM (scene coordinate
    map) video generation.
  - `scripts/infer_coarse2fine_video.py`: coarse-to-fine video generation.
  - `scripts/infer_trellis_img2scene.py`: single image to 3D scene generation
    through TRELLIS.
- **Generative 3D proxies**
  - Sparse structure VAE / DiT and SLAT VAE / DiT for the TRELLIS scene
    representation.
  - Gaussian-splatting scene decoding via the RaDe-GS rasterizer.
  - Depth conditioning from Depth Anything V3, concatenated width-wise into the
    video branch.
- **Video generation backbone**
  - Integration of Wan2.1-Fun-V1.1-14B-Control for camera-trajectory controlled
    video synthesis.
  - Deferred geometry refinement variants for RGB-D and RGB-SCM outputs.
- **Data engine** (`data_engine/`)
  - Scene data curation and preprocessing pipelines for DL3DV, RE10K and the
    spatial-video setting.
  - DINOv2 based filtering and SLAT / sparse-structure encoding utilities.
- **Acceleration**
  - `extensions/vox2seq`: shift-window 3D attention for TRELLIS, installable
    separately.
- **Packaging and docs**
  - `pyproject.toml` with pinned CUDA 12.4 / PyTorch 2.5.1 dependency stack.
  - README with installation, checkpoint download and inference instructions.

### Fixed

- Corrected the scale normalization of generated depth maps.
- Corrected the Depth Anything V3 checkpoint path resolution.
- Relicensed the project under the Apache License 2.0.
- Added a `NOTICE` file listing bundled and derived third-party components.
