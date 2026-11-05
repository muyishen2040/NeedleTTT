# Set up the class Needle source locally

**Start here before DEE-13 (Needle differentiable fast-weight updates) or any task that imports the course framework.**

## Where does the original Needle source go?

There are **two different directories**:

```text
NeedleTTT/                       ← our GitHub repository
├── third_party/
│   ├── README.md                ← tracked explanation
│   └── needle/                  ← YOU copy the complete class checkout here
│       ├── python/
│       │   └── needle/          ← typical Python package in the course checkout
│       ├── src/                 ← if present in your course version
│       └── ...                  ← keep all original course files in place
├── src/needlettt/backend/       ← OUR vector/matrix TTT code, not the class source
│   ├── reference_numpy.py      ← tiny NumPy oracle for correctness
│   ├── needle_ops.py           ← Needle implementation landing zone (DEE-13)
│   └── cuda/                   ← project CUDA integration area
└── scripts/check_needle.py     ← local import diagnostic
```

**Only one recommended location: `NeedleTTT/third_party/needle/`.**

Do not copy the checkout into `src/needlettt/backend/`, and avoid accidentally nesting it as `third_party/needle/needle/python/needle`. The example `python/needle/` layout is common for our course but **not a guarantee**: use the structure in the actual class materials and don't rearrange its files.

This checkout is ignored by Git (`.gitignore` contains `third_party/needle/`), so each teammate needs **their own local copy**. The GitHub repository and PR include **no class Needle source**.

## Step 1 — Copy the existing class checkout

From the root of the NeedleTTT repository, on **Windows PowerShell**:

```powershell
New-Item -ItemType Directory -Force .\third_party | Out-Null
Copy-Item -LiteralPath "C:\path\to\your\course-needle" -Destination ".\third_party\needle" -Recurse
```

Replace `C:\path\to\your\course-needle` with the path to the **entire folder** containing the class Needle sources (not just the inner `needle` Python package). If `third_party/needle` already exists, stop and inspect it instead of copying a second nested folder.

On **macOS/Linux**:

```bash
mkdir -p third_party
cp -R /path/to/your/course-needle third_party/needle
```

These commands assume `third_party/needle` does **not already exist**. Alternatively, use your file manager to copy the entire class checkout into that exact destination.

For the typical course layout, check that you now have:

```text
third_party/needle/python/needle/__init__.py
```

If your class distribution uses a different structure, adapt the package-path instructions in the next step rather than moving files around without understanding its build.

## Step 2 — Check imports (from repository root)

Activate your Python environment and install project dependencies first:

```bash
python -m pip install -e ".[api,dev]"
python scripts/check_needle.py
```

The script adds `third_party/needle/python` to the current Python process's import path and verifies that `import needle` resolves to **your local checkout**, not an unrelated installed package.

- **Success:** it prints the class checkout and imported module path. You can start integrating the real Needle operators in `src/needlettt/backend/needle_ops.py`.
- **Missing directory:** copy the full checkout into `third_party/needle/`.
- **Missing `python/needle` package:** inspect the class checkout for a different layout; coordinate the import path with your teammate.
- **Import fails because a compiled backend is missing:** follow the **build/install instructions in the course's Needle checkout**. The bootstrap cannot guess whether your course version uses CMake, pybind11 or a prebuilt extension.

The check verifies imports only. It does **not** build CUDA, validate gradients, or claim NeedleTTT has been trained.

## Step 3 — If you want to import Needle in an interactive Python session

Setting the Python import path for your current terminal is a lightweight way to use the class source without changing project files.

**Windows PowerShell** (from repository root):

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "third_party\needle\python")
python -c "import needle; print(needle.__file__)"
```

**macOS/Linux** (from repository root):

```bash
PYTHONPATH="$PWD/third_party/needle/python" python -c 'import needle; print(needle.__file__)'
```

For longer-term use, follow the course checkout's documented editable-install instructions if it supports them. Our project deliberately avoids hardcoding an unverified `pip install` or CUDA build command.

## Step 4 — What to implement where

- **Member B:** implement gradient-correct `vector_step`, `matrix_step`, `matrix_scan` in `src/needlettt/backend/needle_ops.py`, using the class's `needle` Python API. Put any project-level CUDA wrapper/kernel in `src/needlettt/backend/cuda/` and use the course backend build approach only when confirmed.
- **Member A:** use those operators from `src/needlettt/models/` and `src/needlettt/training/`. Do not silently switch back to the NumPy teaching model for the final Needle training.
- **Member C:** use the agreed model adapter through FastAPI and keep a CPU fallback.

A modified course backend file may eventually be necessary for GPU support. Before tracking it or providing it to collaborators, confirm the course redistribution rules and project-specific integration plan with the team.

## Important: repository visibility and course rules

At the time of writing, the GitHub repository is **public**. It is safe to copy the class checkout **locally into the ignored `third_party/needle/` directory**, but do not commit or publish the course code or class solutions. Change repository visibility to **private** before including any restricted source, and comply with the course's own rules even in a private repository.

Run `git status --short` before making a PR: no copied class Needle files should appear as tracked changes.
