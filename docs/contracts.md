# Shared contracts (DEE-5) — bootstrap v0

**Source of truth:** [DEE-5 in Linear](https://linear.app/deep-learning-system-final/issue/DEE-5/agree-on-the-shared-model-and-api-interface). Review this with all three teammates before extending the code. The bootstrap is a runnable **single-machine teaching adapter**, not the final trained Needle implementation.

## Data and item identity

- `item_id: str`: external MovieLens movie ID (as a string) or Amazon `parent_asin`. HTTP never exposes model indices.
- `item_idx: int`: contiguous 0-based model index, stable per saved checkpoint/catalog.
- Catalog row: `{item_id, item_idx, title, categories: string[], image_url: string|null, source: string}`.
- Rating event: `{item_id: string, rating: int 1..5, timestamp?: int}`; chronological order is mandatory in dataset replay.

## Python model adapter

```python
state = model.init_state(batch_size=1)
scores = model.recommend(state, candidate_ids)          # [B,C] float, pure/read-only
new_state = model.update(state, item_ids, ratings)     # both inputs [B]
replayed = model.replay(histories)                     # B ordered lists of (idx,rating)
```

- `candidate_ids` are **integer model indices** `[B,C]` and results preserve candidate order.
- `state` keeps a separate observed history and optional vector or matrix fast memory **per user**.
- `update` is nonmutating: it returns new state. All tested variants observe the same rating.
- `replay` starts from fresh initial state and must agree with stepwise updates.
- Variants implemented in the *teaching adapter*: `base`, `vector`, `matrix`. A separately trained `static` and capacity-matched `d²` vector belong to the **final algorithm deliverables**, not the teaching adapter.
- Real model dimensions, trained embeddings, batch/padding semantics and checkpoint serialization are not implemented yet.

## Math that the Needle backend must preserve

Vector: `y=rating-3`, `prediction=dot(w,e)`, `w'=w-eta*(prediction-y)*e/(dot(e,e)+eps)`.

Matrix: `k=normalize(P_key@e)`, `v=((rating-3)/2)*e` and:

```text
error = W @ k - v
W_next = W - eta * outer(error, k) / (dot(k,k) + eps)
memory_adjustment = dot(e, W @ k) / (dot(e,e) + eps)
```

Needle kernel APIs planned: `matrix_step(W,k,v,eta,eps)`, `matrix_scan(W0,keys,values,lengths,eta,eps)`, `vector_step(w,features,y,eta,eps)`. The CUDA path optimizes matrix forward/replay only at first; gradients flow through the **Needle** reference.

## Local HTTP (React → FastAPI)

Base URL: `http://127.0.0.1:8000`, browser on `http://127.0.0.1:5173`. **HTTP does not imply publishing** the service.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Show scaffold/model status |
| POST | `/sessions` | Independent in-memory session |
| GET | `/items?query=...` | Browse the catalog |
| GET | `/sessions/{id}/recommendations?variant=base` | Rank unseen items for one variant |
| POST | `/sessions/{id}/ratings` | `{item_id,rating}`, update all variants |
| POST | `/sessions/{id}/replay` | `{events:[{item_id,rating},...]}`, reset then replay |
| POST | `/sessions/{id}/reset` | Remove history and restore initial state |

Recommendation rows: `{item_id,title,categories,image_url,score,rank}`. The bootstrap additionally returns `model_status:"toy_untrained"`, so it cannot be mistaken for a trained model.

The **final integration** will load actual checkpoints and catalog metadata while keeping item IDs, event timing and response schema compatible. Do not change these boundaries independently.
