# NeedleTTT

**Deep Learning Systems final project:** a history-aware two-tower recommender with optional **vector** or **matrix** test-time training (TTT). We study ranking quality, personalization after ratings, and the cost of fast-weight updates in Needle and CUDA.

[Linear project](https://linear.app/deep-learning-system-final/project/needlettt-9724c36843f8/overview) · [DEE-5 shared contract](https://linear.app/deep-learning-system-final/issue/DEE-5/agree-on-the-shared-model-and-api-interface) · [teammate model tutorial](https://linear.app/deep-learning-system-final/document/needlettt-teammate-guide-two-tower-vector-ttt-matrix-ttt-3f53fcd5c0b6)

> **Bootstrap status:** The runnable sample in this repository uses a **deterministic, untrained NumPy teaching model** with fictional demo products. It is NOT the real GRU two-tower, trained TTT, Needle, or CUDA implementation. Never use its recommendations or scores as experimental evidence.

## Fresh clone and local quickstart (no deployment)

The scaffold is in **[draft PR #1](https://github.com/muyishen2040/NeedleTTT/pull/1)** until the team merges it. Until then, check out the PR branch before running these commands:

```bash
git clone https://github.com/muyishen2040/NeedleTTT.git
cd NeedleTTT
git checkout chore/bootstrap-project-scaffold
```

After the PR has been merged, clone or update `main` normally instead. You do **not** need the class Needle source just to run the toy demo and the bootstrap tests.

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

**Maggie Pai (and anyone running the real Needle model):** Copy your existing course Needle source folder into **`third_party/needle/`** at the root of this repository. Keep the folder's original contents and subdirectories unchanged. Do **not** put the class framework into `src/needlettt/backend/`: that directory is for **our new TTT operators**.

For the common course layout, the resulting path looks like `third_party/needle/python/needle/`. Your exact class checkout may vary; [docs/needle_setup.md](docs/needle_setup.md) has step-by-step Windows and macOS/Linux copy commands, an import check, and troubleshooting.

The course checkout is intentionally **gitignored** and will not appear in GitHub PRs. First verify the class sharing rules; the repository is currently public, so do not push course Needle solutions or other restricted files.

## Project layout

```text
AGENTS.md                  Coding-agent rules; read before editing
docs/                      Architecture, contracts, dataset rules, collaboration
configs/                   Planned dataset configurations (no raw data)
data/examples/             Fictional demo catalog only
src/needlettt/
  data/                    Samuel Shen: MovieLens/Amazon preprocessing
  models/                  Samuel Shen: model contracts + toy reference
  training/                Samuel Shen: end-to-end Needle training
  evaluation/              Samuel Shen: rolling metrics
  backend/                 Maggie Pai: Needle fast-weight ops, CUDA integration
third_party/needle/         LOCAL ONLY: original class Needle checkout (gitignored)
apps/api/                  Pin-Wen Wang: local-only FastAPI service
apps/web/                  Pin-Wen Wang: React/Vite interface
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
- Samuel Shen replaces the teaching model with a trained history-aware two-tower and real data/checkpoints.
- Maggie Pai adds differentiable Needle vector/matrix update operators and benchmarks CUDA fusion.
- Pin-Wen Wang keeps HTTP payloads stable while swapping the teaching model for the trained inference adapter.

Use one branch/PR per Linear ticket. Link the ticket in the PR and run `pytest -q` before requesting review.
