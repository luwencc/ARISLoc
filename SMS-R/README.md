# SMS-R

Sparse path–phase RSSI fingerprinting for reference-point (RP) classification and indoor localization.

## Overview

SMS-R trains a dual-branch **CNN–LSTM–SE** network on a sparse `phase × path` RSS grid:

- **Classification**: 52 RP classes (`RP_LIST` in `smsr/config.py`)
- **Regression**: planar `(x, y)` head fused with Top-K soft RP coordinates at test time

## Repository layout

```
SMS-R/
├── SMSR.py                 # entry point & public import API
├── requirements.txt
└── smsr/
    ├── config.py           # hyperparameters
    ├── model.py            # MixedModel
    ├── normalize.py        # z-score helpers
    ├── data.py             # datasets / caches / grids
    ├── train.py            # training loop
    ├── evaluate.py         # test evaluation & plots
    └── utils.py            # AMP / DataParallel helpers
```

## Requirements

- Python 3.9+
- CUDA-capable GPU recommended

```bash
pip install -r requirements.txt
```

## Data layout

Place next to `SMSR.py` (or set paths in `smsr/config.py`):

```
Trained_Data/<RP>/<RP>_pathN_phaseM/*.csv
Validation_Data/...
Test_Data/...
ordinate.csv
```

Each CSV should contain an `RSSI` column (or a numeric first column). Length must be `segments × 38` (e.g. 152 for four paths).

## Usage

```bash
cd SMS-R
python SMSR.py
```

Set `RETRAIN = False` in `SMSR.py` to evaluate with existing `best_model_DRL.pth`, `mean_DRL.npy`, and `std_DRL.npy`.

Tune batch size, GPUs, and AMP flags in `smsr/config.py`.

## Outputs

| File | Description |
|------|-------------|
| `best_model_DRL.pth` | Best validation checkpoint |
| `mean_DRL.npy` / `std_DRL.npy` | Train z-score stats |
| `Test_picture_DRL/` | Curves, CDF, confusion matrix |

## Import from PPORSLoc

```python
import SMSR as loc
model = loc.MixedModel()
```

Ensure the `SMS-R` directory is on `sys.path` (PPORSLoc does this automatically).

## License

Add your license here before publishing.
