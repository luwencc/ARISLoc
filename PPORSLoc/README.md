# PPORSLoc

PPO-based deep reinforcement learning for **path–phase** selection on a RIS / metasurface localization pipeline.

Uses the frozen **SMS-R** localizer (`SMSR.MixedModel`) as the oracle for localization error.

## Overview

- **Actions**: phase `0…5` (extend next path segment) + **stop**
- **Observation**: flattened sparse phase–path RSS grid + meta features
- **Reward**: MADRL-CIL style (step cost + terminal normalized error)

## Repository layout

```
PPORSLoc/
├── PPORSLoc.py             # CLI entry point
├── requirements.txt
└── pporsloc/
    ├── config.py           # PPO / env hyperparameters
    ├── caches.py           # traj & inference LRU caches
    ├── traj.py             # folder keys / path-phase helpers
    ├── localizer.py        # SMSR inference engine
    ├── env.py              # Gymnasium environment
    ├── policy.py           # Actor-Critic
    ├── ppo.py              # GAE + PPO update
    ├── plots.py            # curves / CDF / stats
    └── train.py            # training loop
```

## Requirements

```bash
pip install -r requirements.txt
```

Also install and train **SMS-R** first so these files exist (paths in `pporsloc/config.py`):

- `best_model_DRL.pth`
- `mean_DRL.npy` / `std_DRL.npy`
- `ordinate.csv`

## Data

Default root is `Test_Data` in config; for training prefer a `DRL_RP`-style tree:

```
DRL_RP/<RP>/<RP>_path1_phase{p}_path2_phase{q}_.../*.csv
```

```bash
python PPORSLoc.py --drl-root /path/to/DRL_RP
```

## Usage

```bash
cd PPORSLoc
python PPORSLoc.py --device cuda:0 --episodes 2500
```

Useful flags: `--synthetic`, `--cache-gb`, `--loc-batch`, `--eval-greedy`.

## Dependency on SMS-R

PPORSLoc imports the sibling package:

```python
import SMSR as loc
```

Keep this layout when uploading to GitHub:

```
GITHUB/
├── SMS-R/
│   └── SMSR.py
└── PPORSLoc/
    └── PPORSLoc.py
```

## Outputs

| File | Description |
|------|-------------|
| `drl_ppo_path_phase.pt` | Trained policy |
| `drl_ppo_train_curves_panel.png` | Training curves |
| `drl_ppo_error_cdf.png` | Eval CDF |
| `drl_ppo_episode_returns.csv` | Episode logs |

## License

Add your license here before publishing.
