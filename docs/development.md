# Development and review workflow

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

## Sensitive/course code

The repository was created **public**. Do not commit course Needle solutions, credentials, datasets, checkpoint weights or personal data. Ask the owner to switch visibility to **private** before sharing restricted course implementations; verify course-specific distribution rules first. Real data and training outputs live only under gitignored directories.

## Hand-off checklist

- `pytest -q` passes locally or clearly explain blockers.
- For frontend changes, `npm run build` passes.
- Confirm new file paths and interfaces in the relevant Linear issue.
- Include minimal reproducible commands and expected behavior.
