# NeedleTTT

**Deep Learning Systems final project:** a history-aware two-tower recommender with optional **vector** or **matrix** test-time training (TTT). We study ranking quality, personalization after ratings, and the cost of fast-weight updates in Needle and CUDA.

[Linear project](https://linear.app/deep-learning-system-final/project/needlettt-9724c36843f8/overview) · [DEE-5 shared contract](https://linear.app/deep-learning-system-final/issue/DEE-5/agree-on-the-shared-model-and-api-interface) · [teammate model tutorial](https://linear.app/deep-learning-system-final/document/needlettt-teammate-guide-two-tower-vector-ttt-matrix-ttt-3f53fcd5c0b6)

> **Bootstrap status:** The runnable sample in this repository uses a **deterministic, untrained NumPy teaching model** with fictional demo products. It is NOT the real GRU two-tower, trained TTT, Needle, or CUDA implementation. Never use its recommendations or scores as experimental evidence.

## Local quickstart (no deployment)

Requires Python 3.11+ and Node 20+.

```bash
python -m venv .venv
# activate .venv using your OS-specific command
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
apps/api/                  Member C: local-only FastAPI service
apps/web/                  Member C: React/Vite interface
tests/                     Shared correctness + API smoke tests
scripts/                   Experiments, benchmarks, data entrypoints
```

**Do not copy or reorganize the course Needle source into this public repository.** Its imports, license and assignment-sharing rules must be checked first. The repository currently contains only our scaffolding and independently written reference code. We recommend changing the repo to **private before adding course implementation files**.

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
