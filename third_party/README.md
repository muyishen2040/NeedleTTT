# Local course framework checkout

Copy the **entire original course Needle checkout** into **`third_party/needle/`** on your local machine.

Example: `third_party/needle/python/needle/` (layout varies with course release).

The folder `third_party/needle/` is intentionally gitignored, so this tracked README is the only file that appears in GitHub. Every teammate needs their own local copy.

Step-by-step Windows/macOS/Linux instructions: [docs/needle_setup.md](../docs/needle_setup.md). Verify import from the repository root with:

```bash
python scripts/check_needle.py
```

Do not copy course sources into `src/needlettt/backend/`; that directory is for this project's new TTT operators. Do not commit restricted course code.
