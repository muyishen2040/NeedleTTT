# React/Vite local teaching UI

The UI shows fictional games and calls a **local** Python API. No hosting needed.

```bash
# First run the API from the repository root.
cd apps/web
npm install
npm run dev
```

Open http://127.0.0.1:5173. The default backend is `http://127.0.0.1:8000` (override with `VITE_API_BASE` at development time if necessary).

**Everything is untrained demo scaffolding.** Product images and trained recommendations will come later from Amazon Video Games metadata and the Needle-trained checkpoint. Keep the DEE-5 HTTP payloads stable when swapping the backend.
