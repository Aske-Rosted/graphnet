# Vendored code: Maxwell Nakos' production bundle models

Deployment-only compatibility layer, so that one GraphNeT installation runs both the
ATHENA multitask reconstruction and Maxwell Nakos' bundle classifiers in the same
IceTray process (his `Selection-Processing-bundle-v3`: `EHE_selection.py` adds our
reconstruction (`step_3`) and his models (`step_2`) to one `I3Tray`).

## Source

GraphNeT commit `99f030bfd4c396c2f5127b1396e58f4e4048dc1b` (2026-06-17): the code in the
production container `lowe_v4.sif`, part of the history of `ZeptoByte/graphnet`
(`multitask_with_bundle`). Class bodies are copied as exact line ranges, only the imports
differ; the files are excluded from the pre-commit hooks so they stay comparable to the
source. Do not edit them: re-vendor instead.

| here | source (`src/graphnet/...` at 99f030bf) |
|---|---|
| `models/vendored/bundle.py`: `IceCubeBundleAdvanced` | `models/detector/icecube.py` |
| `NodesAsPulsesBundle` | `models/data_representation/graphs/nodes/nodes.py` |
| `LateralSpreadReconstruction` | `models/task/reconstruction.py` |
| `RMSEAdjustedLoss`, `FocalCrossEntropyLoss` | `training/loss_functions.py` |
| `data/extractors/icecube/vendored/i3pulseextractornew.py` | `data/extractors/icecube/i3pulseextractornew.py` |
| `data/extractors/icecube/vendored/i3inferencepulseextractornew.py` | `data/extractors/icecube/i3inferencepulseextractornew.py` |
| `data/extractors/icecube/vendored/pulselabeling.py` | `data/extractors/icecube/utilities/pulselabeling.py` |

## Scope

Only what the production models need: the configs under
`/cvmfs/icecube.opensciencegrid.org/users/mnakos/Evaluation_Models{,_2}` used by
`EHE_selection.py` (EHE, point-source and rms-only model sets, and
`Evaluation_Models_2/5_Feature_Classifier` with the new extractor). The extractors are
exported from `graphnet.data.extractors.icecube` under their original names.

Outside these files (regular, linted code):
- `CrossEntropyLoss(ratio=...)`: first-class weight; his configs pass `ratio: 1` and his
  state dicts hold the weight buffer.
- `I3MultipleModelInferenceModule(first_column_only=True)`: the production map layout
  (`<key_name>[<model_name>]` = first prediction column, e.g. the `neutrino` softmax
  probability); `multiple_models=True` is accepted as a deprecated alias.
- `I3InferenceModule(num_threads=..., inference_speed_check=...)`: the production
  keyword arguments (`inference_speed_check` = `record_runtimes`).

## Removing it

Once these models load with an upstream-based GraphNeT (Maxwell's work, his call),
delete both `vendored/` folders, their export in `data/extractors/icecube/__init__.py`
and the `vendored/` exclusion in `.pre-commit-config.yaml`.
