# ARP-CBCT Model

This repository contains the model-only PyTorch implementation of:

**ARP-CBCT: Adaptive Ray Primitive Representation for Sparse-view CBCT Reconstruction**

The released architecture is the complete `arp_cbct_v4_balanced` model used for the reported 20-, 10-, and 5-view experiments. Training, evaluation, data-preparation code, checkpoints, and private clinical data are intentionally not included at this stage. The complete implementation will be made publicly available upon acceptance of the paper.

## What Is Included

- Shared multi-scale 2D projection encoder.
- Geometry-conditioned feature modulation.
- Adaptive, ray-grounded primitive proposal with multiple depth hypotheses.
- Multi-view geometry query and geometry-aware attention.
- Local primitive interaction and bounded primitive refinement.
- Continuous anisotropic primitive splatting at coarse and intermediate scales.
- Balanced hierarchical 3D decoder and physics-residual fusion.
- Vector-based cone-beam projection geometry utilities required by the model forward pass.
- The exact model configuration used for the reported experiments.

## Repository Structure

```text
arp_cbct/
├── __init__.py
├── config.py
├── geometry.py
├── model.py
└── models/
    ├── arp_cbct_v4_balanced.py
    ├── arp_cbct_v3.py
    ├── arp_cbct_v2.py
    ├── encoder2d.py
    ├── geometry_conditioning.py
    ├── primitive_proposal_v2.py
    ├── primitive_query.py
    ├── primitive_attention.py
    ├── primitive_interaction.py
    ├── primitive_refinement_v2.py
    ├── primitive_splatting.py
    └── hierarchical_decoder_balanced.py
```

## Dependency

PyTorch is required.

```bash
pip install -r requirements.txt
```

The reported experiments used Python 3.10 and PyTorch 2.0.0 with CUDA 11.8.

## Model Construction

```python
from arp_cbct import ARPCBCT, get_default_config

config = get_default_config()
model = ARPCBCT(config)
```

The default configuration constructs the exact model variant used for the formal experiments, with 8,192 adaptive primitives and a multi-scale reconstruction hierarchy.

## Forward Interface

```python
outputs = model(
    projections=projections,
    view_mask=view_mask,
    geometry=geometry,
    grid=reconstruction_grid,
)
prediction = outputs["volume"]
```

Inputs follow these conventions:

- `projections`: `[B, V, 1, H, W]` line-integral projections.
- `view_mask`: `[B, V]` validity mask, enabling a variable number of views.
- `geometry`: `ProjectionGeometry` containing per-view source, detector-center, detector-u, and detector-v vectors in the common world frame.
- `reconstruction_grid`: `ReconstructionGrid` describing the output grid in physical coordinates.

The primary output `outputs["volume"]` has shape `[B, 1, D, H, W]`. The returned dictionary also exposes primitive positions, supports, confidences, attention weights, physics density, decoder residuals, and multi-scale coverage for research analysis.

## Geometry Convention

- Volume arrays use `z-y-x` indexing.
- World-coordinate vectors use `x-y-z` ordering and millimetres.
- Detector `u` and `v` vectors represent one native detector-pixel step in world coordinates.
- Projection geometry and the reconstruction grid must already be expressed in the same coordinate system.
- Missing scanner geometry must not be replaced with arbitrary default values.

## Notes for Training Integration

This repository does not currently include:

- data loaders or private data;
- preprocessing and scanner-to-reference registration code;
- reconstruction or projection-consistency losses;
- training, validation, or testing scripts;
- checkpoints or experiment logs.

These components will be released after paper acceptance. The current repository is intended to disclose the complete model construction and its geometry-aware forward interface.

## Citation

The citation will be added when the manuscript becomes publicly available.

## Contact

For questions, please open a GitHub issue.
