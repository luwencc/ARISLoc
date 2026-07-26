# Data

Large training / validation / test archives are **not** stored in this Git folder (files are too big for normal git).

## Download

Get the datasets from the repository **Releases** page:

**https://github.com/luwencc/PPORSLoc/releases/tag/Data**

Or open the repo → right sidebar **Releases** → release **Dataset** (tag `Data`).

## Files

| Asset | Description |
|-------|-------------|
| `Trained_Data.tar.zip` | Training set for SMS-R |
| `Validation_Data.tar.zip` | Validation set for SMS-R |
| `Test_Data.tar.zip` | Test set for SMS-R / PPORSLoc |

## Setup

1. Download the archives from the release above.
2. Extract them so you get folders such as `Trained_Data/`, `Validation_Data/`, and `Test_Data/`.
3. Place those folders where the code expects them (by default next to `SMSR.py` / paths in `SMS-R/smsr/config.py`), or update the paths in config.

You also need `ordinate.csv` for coordinate labels if it is not already present in your working directory.
