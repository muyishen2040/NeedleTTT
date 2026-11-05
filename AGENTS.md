# Instructions for AI coding agents

This repo is a **three-person course research project**, not a production storefront. Keep changes small, understandable, and consistent with the shared interfaces. Read the assigned Linear ticket, then these files:

1. `docs/architecture.md` — ownership and data flow.
2. `docs/contracts.md` — Python, tensor and local HTTP interfaces.
3. `docs/data.md` — the no-leakage rolling evaluation protocol.
4. `docs/development.md` — commands and PR checklist.
5. `docs/needle_setup.md` — **the canonical path and import-check procedure for the class Needle source**.

## Ownership boundaries

- **Algorithm / Member A:** `src/needlettt/data/`, `models/`, `training/`, `evaluation/`, `configs/`.
- **Needle/CUDA / Member B:** `src/needlettt/backend/`, its tests and benchmarks. The original class framework belongs **locally** at `third_party/needle/` (gitignored); retain its original layout and follow `docs/needle_setup.md`. Do not vendor or commit the course checkout.
- **Frontend / Member C:** `apps/api/`, `apps/web/`, their tests.
- **Shared:** `src/needlettt/contracts.py`, `docs/contracts.md`, package boundaries. Ask teammates for review before changing public signatures.

## Non-negotiable correctness rules

- Browser/API use **external string `item_id`**; model uses saved **integer `item_idx`**.
- A rating cannot influence its own pre-feedback prediction. Run **predict → measure → reveal → update**.
- Ordinary two-tower recomputes its user embedding with new history; both TTT variants see the **same** feedback. Shared model weights remain fixed at inference.
- Per-user fast weights must not leak between sessions. Updates return new states during differentiated training.
- Clearly label **untrained demo data/model**. No fictional rankings masquerading as research results.
- Final accuracy numbers come from held-out users, the declared catalog, documented seeds, and paired uncertainty when possible.
- Do not commit real Amazon data, checkpoints, secrets, user information, or restricted course Needle solutions.
- Keep a CPU path. No web hosting, cloud services, DB or auth are required for the local demo.

## Editing workflow

1. Read the ticket acceptance criteria and inspect neighboring code before writing.
2. Prefer existing modules; avoid duplicate interfaces, new dependencies or broad rewrites.
3. Add a test demonstrating behavior or the bug. Mark unimplemented model/backend parts honestly.
4. Run `pytest -q`; for web changes run `npm run build` inside `apps/web`.
5. In PR summary, link the Linear issue, list changed files, commands run, model/data assumptions and remaining limitations.
6. Avoid editing another member's directory unless the shared contract requires it; coordinate changes in `docs/contracts.md`.

The NumPy teaching model in `models/reference_numpy.py` exists to exercise API wiring and provide a **small numerical oracle**. It is not an acceptable substitute for end-to-end Needle training.
