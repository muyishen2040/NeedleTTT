# NeedleTTT

**Deep Learning Systems final project:** a history-aware two-tower recommender with optional **vector** or **matrix** test-time training (TTT). We study ranking quality, personalization after ratings, and the cost of fast-weight updates in Needle and CUDA.

[Linear project](https://linear.app/deep-learning-system-final/project/needlettt-9724c36843f8/overview) · [DEE-5 shared contract](https://linear.app/deep-learning-system-final/issue/DEE-5/agree-on-the-shared-model-and-api-interface) · [teammate model tutorial](https://linear.app/deep-learning-system-final/document/needlettt-teammate-guide-two-tower-vector-ttt-matrix-ttt-3f53fcd5c0b6)

> **Bootstrap status:** The runnable sample in this repository uses a **deterministic, untrained NumPy teaching model** with fictional demo products. It is NOT the real GRU two-tower, trained TTT, Needle, or CUDA implementation. Never use its recommendations or scores as experimental evidence.

## Shared Google Drive + Colab workflow (DEE-30)

For team experiments, keep **code/notebooks in GitHub** and **datasets, checkpoints and result artifacts in one shared Google Drive folder**. See [the shared Drive + Colab setup guide](docs/colab_workflow.md) and the [starter Colab notebook](notebooks/00_setup.ipynb). The default shared path is `/content/drive/MyDrive/NeedleTTT`, with `DATA_ROOT=/content/drive/MyDrive/NeedleTTT/data`; each teammate must add a shortcut to the *same* shared folder, not create separate copies.

**Setup status:** The documentation/notebook alone do not verify sharing permissions, actual dataset availability, or successful teammate runs. Track remaining checks in [DEE-30](https://linear.app/deep-learning-system-final/issue/DEE-30/set-up-shared-google-drive-and-colab-workspace).

## Fresh clone and local quickstart (no deployment)

The [bootstrap PR #1](https://github.com/muyishen2040/NeedleTTT/pull/1) is merged into `main`. Start from the current default branch:

```bash
git clone https://github.com/muyishen2040/NeedleTTT.git
cd NeedleTTT
```

You do **not** need the class Needle source just to run the toy demo and bootstrap tests.

## Local quickstart (no deployment)

Requires Python **3.11+** and Node **20+**. Run these commands in the repository root.

- **Windows PowerShell:** activate the Python environment with `.\\.venv\\Scripts\\Activate.ps1`.
- **macOS/Linux:** activate it with `source .venv/bin/activate`.

Python package and toy demo work independently of the class Needle checkout. For the real Needle model, follow [docs/needle_setup.md](docs/needle_setup.md).

```bash
python -m venv .venv
# activate the venv using your OS-specific command above
python -m pip install -e ".[api,dev]"
pytest -q
uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

In a **second terminal**:

```bash
cd apps/web
npm install
npm run dev
```

Open http://127.0.0.1:5173. Everything runs locally; **no cloud deployment, database, or user account is required**. The API listens on loopback (`127.0.0.1`) only.

## Set up the course Needle checkout

**Member B (and anyone running the real Needle model):** Copy your existing course Needle source folder into **`third_party/needle/`** at the root of this repository. Keep the folder's original contents and subdirectories unchanged. Do **not** put the class framework into `src/needlettt/backend/`: that directory is for **our new TTT operators**.

For the common course layout, the resulting path looks like `third_party/needle/python/needle/`. Your exact class checkout may vary; [docs/needle_setup.md](docs/needle_setup.md) has step-by-step Windows and macOS/Linux copy commands, an import check, and troubleshooting.

The course checkout is intentionally **gitignored** and will not appear in GitHub PRs. First verify the class sharing rules; the repository is currently public, so do not push course Needle solutions or other restricted files.

## Project layout

```text
AGENTS.md                  Coding-agent rules; read before editing
docs/                      Architecture, contracts, dataset rules, collaboration
configs/                   Planned dataset configurations (no raw data)
data/examples/             Fictional demo catalog only
src/needlettt/
  data/                    Member A: MovieLens/Amazon preprocessing
  models/                  Member A: model contracts + toy reference
  training/                Member A: end-to-end Needle training
  evaluation/              Member A: rolling metrics
  backend/                 Member B: Needle fast-weight ops, CUDA integration
third_party/needle/         LOCAL ONLY: original class Needle checkout (gitignored)
apps/api/                  Member C: local-only FastAPI service
apps/web/                  Member C: React/Vite interface
tests/                     Shared correctness + API smoke tests
scripts/                   Experiments, benchmarks, data entrypoints
```

**Local copy is supported; committing it is not.** The course Needle source may be copied locally under `third_party/needle/`, which is ignored by Git. Check the course sharing rules before considering any redistribution. Change the repository to **private** before you add any restricted course work to tracked files. See [Needle setup](docs/needle_setup.md).

## Model and evaluation plan

1. History-updated neural two-tower (GRU user encoder).
2. Two-tower + vector TTT (including a capacity-matched variant).
3. Two-tower + matrix TTT with rating-conditioned vector targets.
4. Separately trained static/no-update control.

MovieLens-1M is the development dataset; Amazon Reviews 2023 Video Games 5-core is the primary benchmark and UI catalog; Office Products is a secondary replication. Use **predict → evaluate → reveal rating → update** per event, with NDCG@10 over the declared full *eligible/filtered* catalog. See [docs/data.md](docs/data.md) for leakage and reporting rules.

## First team tasks

- Review [docs/contracts.md](docs/contracts.md) together before coding and confirm DEE-5.
- Member A replaces the teaching model with a trained history-aware two-tower and real data/checkpoints.
- Member B adds differentiable Needle vector/matrix update operators and benchmarks CUDA fusion.
- Member C keeps HTTP payloads stable while swapping the teaching model for the trained inference adapter.

Use one branch/PR per Linear ticket. Link the ticket in the PR and run `pytest -q` before requesting review.
