# Shared Google Drive + Colab workflow (DEE-30)

This guide standardizes how NeedleTTT teammates share **datasets and experiment outputs**, while keeping **code and notebooks in GitHub**. It is setup guidance, **not confirmation** that a Drive folder is shared or accessible to everyone.

- Linear: [DEE-30 — Shared Drive and Colab setup](https://linear.app/deep-learning-system-final/issue/DEE-30/set-up-shared-google-drive-and-colab-workspace)
- Related: [DEE-6 data preparation](https://linear.app/deep-learning-system-final/issue/DEE-6/prepare-movielens-and-amazon-rating-histories), [DEE-8 model training](https://linear.app/deep-learning-system-final/issue/DEE-8/train-neural-two-tower-with-vector-and-matrix-ttt), [DEE-27 reproducibility](https://linear.app/deep-learning-system-final/issue/DEE-27/package-reproducible-colab-source-and-experiment-outputs)
- Runnable starter: [notebooks/00_setup.ipynb](../notebooks/00_setup.ipynb)

## 1. Shared Drive folder and permissions

1. One teammate creates (or identifies) a Google Drive folder named `NeedleTTT` and shares its **actual folder** with the other two teammates with Editor access. Agree on who owns the folder and avoid duplicating it.
2. Each teammate opens the shared folder from **Shared with me** and chooses **Organize → Add shortcut → My Drive** (UI wording may vary). Name the shortcut exactly `NeedleTTT` and place it at the top level of **My Drive**.
3. Each teammate checks that this shortcut points to the *same shared folder*, not a newly created folder with the same name. Test by creating and reading a harmless file with the team's agreement.
4. Do not publish the shared folder link or add teammates' email addresses, account credentials, private file IDs, or secrets to this public repo. Share the link directly with your teammates.

Google Colab mounts the signed-in user's Drive, **not other users' Drive namespaces**. Matching shortcut paths offer a consistent pathname without copying all data three times. Permissions must still be checked independently by each account.

## 2. Canonical folder layout

Create the following *inside the shared Drive folder*, not in GitHub:

```text
My Drive/
└── NeedleTTT/                  # shortcut to ONE shared folder
    ├── data/
    │   ├── raw/                # source archives; read-only for downstream notebooks
    │   └── processed/          # immutable versioned datasets/splits/catalogs
    ├── checkpoints/            # per-run model artifacts
    └── results/                # per-run metrics, plots and metadata
```

Conventions:
- `data/processed/<dataset>/<version>/` should include a stable catalog mapping, user splits, events, and `manifest.json` (source, filtering rules, seed, data counts, version).
- Save training artifacts under `checkpoints/<model>/<run_id>/`; save evaluation outputs under `results/<run_id>/`. Use unique run IDs so teammates never overwrite each other's output.
- Never modify published processed data in place; publish a new version and update your run config instead.
- Commit small configs, scripts, and notebooks to GitHub. Keep large data, training outputs, checkpoints, and restricted course Needle source out of Git.

## 3. Colab paths (identical across teammates)

Mount Drive using the starter notebook (or manually):

```python
from google.colab import drive
drive.mount("/content/drive")
```

The agreed default is:

```python
from pathlib import Path
import os

DRIVE_ROOT = Path(os.environ.get(
    "NEEDLETTT_DRIVE_ROOT",
    "/content/drive/MyDrive/NeedleTTT",
))
DATA_ROOT = DRIVE_ROOT / "data"
CHECKPOINT_ROOT = DRIVE_ROOT / "checkpoints"
RESULTS_ROOT = DRIVE_ROOT / "results"
os.environ["DATA_ROOT"] = str(DATA_ROOT)
```

**`DATA_ROOT` is `/content/drive/MyDrive/NeedleTTT/data` by default.** If a teammate cannot create the shortcut at this exact location, set `NEEDLETTT_DRIVE_ROOT` *before executing the path cell* and document the per-user override privately; do not hardcode a different path in each training notebook.

Never silently `mkdir` a missing `DRIVE_ROOT`: it could create a **private copy** rather than opening the actual shared folder. The starter checks expected directories and fails with actionable instructions.

## 4. Starter notebook and GitHub

Open [00_setup.ipynb](../notebooks/00_setup.ipynb) in Google Colab. If using the Colab CLI, run `colab drivemount -s <session>` first; the notebook reuses an existing Drive mount. It:
1. mounts Google Drive;
2. resolves and validates Drive paths;
3. clones the current GitHub `main` branch into the Colab VM if missing;
4. installs the Python package and imports `needlettt`;
5. prints the Git commit and loads the **fictional demo catalog** for a safe import smoke test.

`00_setup.ipynb` does **not** download MovieLens/Amazon, create the shared folder, or train a real model. The repo's default branch is currently public; if it becomes private, configure authenticated access using Colab secrets or an approved credential workflow. **Never paste tokens into a notebook or commit credentials.**

To obtain the three raw datasets after setup, run [01_download_datasets.ipynb](../notebooks/01_download_datasets.ipynb). It downloads the two Amazon 5-core rating files and separate metadata to the shared `data/raw/amazon_video_games/` and `data/raw/amazon_grocery/` folders, using about 2.1 GB. Confirm the data's sharing terms before making it available to teammates. MovieLens-1M is downloaded and extracted only to `/content/needlettt-local-data/movielens_1m/`, because GroupLens prohibits redistribution without separate permission; runtime files disappear after a reset. The downloader verifies size and SHA-256 and will not overwrite a mismatched existing file.

This notebook only downloads raw inputs. It does not create the processed events, catalog, splits, and manifests required by DEE-6. No raw datasets are committed to GitHub.

Because Colab VM storage is temporary, a repo clone under `/content/NeedleTTT` will disappear when the runtime resets. Keep source in GitHub and artifacts in Drive. Prefer a local VM copy of large data for training speed, and write final outputs back into versioned shared Drive paths.

## 5. Validation / handoff

Before DEE-30 is considered done, **each teammate** should:
- Confirm Editor access to the same Drive folder (by accessing one agreed test file).
- Mount Drive and confirm `data/raw`, `data/processed`, `checkpoints`, and `results` are visible.
- Run the starter notebook from their **own** Colab account and report the resolved `DATA_ROOT`, Git commit and import result (not private Drive identifiers).
- Coordinate with Samuel ([DEE-6](https://linear.app/deep-learning-system-final/issue/DEE-6/prepare-movielens-and-amazon-rating-histories)) before using a processed dataset: record its manifest/version, item-ID mapping, train/validation/test split and coverage. No real processed datasets are supplied by this setup notebook.

For GPU/Needle/CUDA experiments, record the Colab runtime, framework/CUDA/compiler versions, GPU model, random seeds and Git commit with each result. Follow [Needle setup](needle_setup.md) and the course's sharing rules; the course framework is intentionally not committed.
