# Indoor Localization: SMS-R & PPORSLoc

This repository contains two complementary projects for RIS / metasurface-aided indoor localization from RSSI fingerprints.

| Project | Role |
|---------|------|
| [SMS-R](SMS-R/) | Supervised CNN–LSTM–SE localizer (RP classification + xy) |
| [PPORSLoc](PPORSLoc/) | PPO agent that chooses path/phase measurement sequences |

## Data

Training / validation / test archives are **not** stored in git (each file is tens–hundreds of MB).  
Download them from the repository **[Releases](https://github.com/luwencc/PPORSLoc/releases)** page:

| Archive | Typical use |
|---------|-------------|
| `Trained_Data.tar.xz` | SMS-R training set |
| `Validation_Data.tar.xz` | SMS-R validation set |
| `Test_Data.tar.xz` | SMS-R / PPORSLoc test set |

Extract into the project root (or paths set in `smsr/config.py`), for example:

```bash
# from the repository root
tar -xf Trained_Data.tar.xz
tar -xf Validation_Data.tar.xz
tar -xf Test_Data.tar.xz
```

You also need `ordinate.csv` next to the training scripts (include it with the release assets if it is not already in the repo).

## Quick start

1. Train the localizer:

```bash
cd SMS-R
pip install -r requirements.txt
python SMSR.py
```

2. Train the PPO policy (needs SMS-R weights + DRL trajectory data):

```bash
cd ../PPORSLoc
pip install -r requirements.txt
python PPORSLoc.py --drl-root /path/to/DRL_RP --device cuda:0
```

## Relationship

```
RSS CSV ──► SMS-R (MixedModel) ──► localization error / coords
                ▲
                │ frozen oracle
PPORSLoc (PPO) ─┘ chooses next phase or stop
```

## License

Add a license before publishing to GitHub.
