# Development and review workflow

## Fresh clone

[Bootstrap PR #1](https://github.com/muyishen2040/NeedleTTT/pull/1) has been merged. Clone `main`:

```bash
git clone https://github.com/muyishen2040/NeedleTTT.git
cd NeedleTTT
```

No course Needle source is needed for the teaching demo or bootstrap tests.

## Shared Google Drive / Colab

For team-hosted notebooks, shared datasets, checkpoints, and results, follow [the Colab workflow guide](colab_workflow.md) and [starter notebook](../notebooks/00_setup.ipynb). Source and notebooks remain in GitHub, large artifacts remain in shared Drive, and `DATA_ROOT` defaults to `/content/drive/MyDrive/NeedleTTT/data`. Each teammate must validate access to the **same** folder. See [DEE-30](https://linear.app/deep-learning-system-final/issue/DEE-30/set-up-shared-google-drive-and-colab-workspace).

## Quick commands

```bash
python -m pip install -e ".[api,dev]"
pytest -q
uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
# separate terminal:
cd apps/web && npm install && npm run dev
# optional frontend check:
npm run build
```

This is a **local-only** demo. The API keeps sessions in memory; restarting it clears them.

## Branches and PRs

- Create a branch named after the Linear ticket, e.g. `dee-13/needle-fast-update`.
- Keep the PR scoped to one deliverable. Link the ticket in PR description.
- A PR must state: *What changed? How tested? Contract implications? What is intentionally unfinished?*
- Existing shared contracts are cross-team boundaries. Don't break them just to make a local test pass.
- Prefer smoke tests using the fictional catalog rather than downloading large datasets on CI.
- Do not mark Linear tickets complete until the actual acceptance tests work.

## Preparing the class Needle framework

Before starting DEE-13, follow [Needle local setup](needle_setup.md): copy the **complete** class checkout into `third_party/needle/` (ignored by Git), then run `python scripts/check_needle.py` from the repo root. The check verifies your local import, not the actual TTT operations. Follow your course version's own build instructions if compiled extensions are required. Keep project-specific operations in `src/needlettt/backend/`.

## Sensitive/course code

The repository was created **public**. Do not commit course Needle solutions, credentials, datasets, checkpoint weights or personal data. Ask the owner to switch visibility to **private** before sharing restricted course implementations; verify course-specific distribution rules first. Real data and training outputs live only under gitignored directories.

## Hand-off checklist

- `pytest -q` passes locally or clearly explain blockers.
- For frontend changes, `npm run build` passes.
- Confirm new file paths and interfaces in the relevant Linear issue.
- Include minimal reproducible commands and expected behavior.
