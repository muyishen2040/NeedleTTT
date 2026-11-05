# Local FastAPI adapter

This adapter is **deliberately running an untrained NumPy teaching model** using fictional catalog items. Its purpose is to exercise the DEE-5 HTTP contract while the algorithm and Needle work continues.

```bash
# from repository root:
python -m pip install -e ".[api,dev]"
uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

Interactive docs: http://127.0.0.1:8000/docs. It stores sessions in Python memory; restart the process to clear them. **It does not need deployment.**

When a real checkpoint is available, replace `MODELS`/catalog initialization in `main.py` with the shared inference adapter, preserving external string `item_id` and each endpoint's JSON schema. Do not present teaching scores as learned recommendations.
