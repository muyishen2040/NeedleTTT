# Member B — Needle/CUDA work area

Start with [DEE-13](https://linear.app/deep-learning-system-final/issue/DEE-13) and `docs/contracts.md`.

- `reference_numpy.py`: numerical truth for the **inner** vector/matrix updates; **not** training.
- `needle_ops.py`: clear NotImplementedError placeholders for differentiable Needle functions.
- `cuda/`: landing zone for fused **forward/replay** source and benchmarks, after confirming the course framework's layout.

**First deliverable:** implement Needle `vector_step`, `matrix_step` and `matrix_scan`, test their values *and gradients*, and preserve per-user state isolation. Next optimize matrix replay via CUDA. Avoid copying the course Needle solution into this public repo; the repository should be made private first, following course rules.
