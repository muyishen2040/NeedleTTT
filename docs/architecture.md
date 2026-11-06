# Architecture and ownership

## Goal

Compare **history-updated two-tower recommendation** against two extensions that learn *temporary user-specific parameters* after every observed rating. Implement the training path in Needle and optimize the matrix replay scan in CUDA.

```text
MovieLens / Amazon review histories (Samuel Shen)
       ↓ external item_id → stable item_idx
Neural two-tower + vector/matrix TTT (Samuel Shen)
       ↓ same model protocol for each variant
Needle differentiable vector/matrix operators (Maggie Pai)
       ↓ optional CUDA-accelerated matrix replay
Local FastAPI inference adapter (Pin-Wen Wang)
       ↕ HTTP + JSON on 127.0.0.1
React/Vite browsing and rating UI (Pin-Wen Wang)
```

## Model boundary

A conventional user-history encoder produces `u_t` from previously observed ratings, and the item encoder produces `e_i`. A score starts as `dot(u_t,e_i)+b_i`. A vector fast memory adds `lambda*dot(w_t,e_i)`; a matrix fast memory adds `lambda*dot(e_i,W_t@k_i)/(dot(e_i,e_i)+eps)`.

Fast memory learns by inner squared-error steps after feedback. Shared towers, key projection, fast initialization, learning rate and gate are trained **by outer future-ranking loss** through those updates (truncated BPTT). At inference, freeze shared parameters.

Matrix target `v=((rating-3)/2)*e_i` is **constructed** from the scalar feedback, not observed in the dataset.

## Three independent workstreams

| Workstream | Primary files | First deliverable |
| --- | --- | --- |
| Algorithm | `src/needlettt/{data,models,training,evaluation}` | MovieLens histories + trained two-tower and update variants |
| Needle/CUDA | `src/needlettt/backend` | Differentiable Needle scan matching NumPy oracle |
| Frontend | `apps/api`, `apps/web` | Local UI driven by shared model contract |

Common changes (tensor signatures, item IDs, checkpoint schema, HTTP payloads) require joint review. Start with [DEE-5](https://linear.app/deep-learning-system-final/issue/DEE-5/agree-on-the-shared-model-and-api-interface).

## What exists in the bootstrap

- NumPy **teaching** fast-weight math and toy scoring, with unit tests.
- In-memory FastAPI sessions and local React UI to exercise the future integration boundary.
- Architecture and data contracts, not trained recommender weights or actual Needle/CUDA implementations.

Real MovieLens/Amazon preprocessing, GRU training, Needle autodiff and CUDA fusion are ticket work, not part of this bootstrap.
