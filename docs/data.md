# Datasets and evaluation protocol

## Planned datasets

- **MovieLens-1M:** development and correctness benchmark. Movie titles/genres; useful for fast smoke tests.
- **Amazon Reviews 2023 / Video Games 5-core:** primary held-out ranking benchmark and product UI. Review `parent_asin` is an external item ID. Get titles, categories and image URLs from a **separate** Amazon product metadata file joined on `parent_asin`.
- **Office Products 5-core:** secondary replication using the *same* pipeline and metrics.
- **Grocery:** optional, not a required deliverable.

**No real datasets have been committed.** The `data/examples/catalog.json` file is fictional and solely tests UI and API wiring.

## Split and filtering

1. Split users into 70% train / 15% validation / 15% test using a recorded seed. No overlap.
2. Sort each person's events by timestamp, then external item ID to break ties.
3. If using a 2,000-item Amazon subset, choose items from **training users only**. Record selected item IDs, original and retained user/event counts, and coverage.
4. Save one stable `item_id ↔ item_idx` mapping for training, inference and frontend.
5. Train/tune on train/validation only. Never use test ratings to select models or item features.

## Rolling (prequential) ranking

At each event `t`:

1. Build current state using **only** events before `t`.
2. Rank every eligible candidate in the **declared evaluation catalog**, excluding seen items under the specified non-repeat protocol.
3. If the event has a rating of 4–5 and its target is eligible, compute next-positive NDCG@10/Hit@10.
4. Reveal the event's actual rating (1–5), then update *all* models' histories and fast weights.
5. Continue to the next event. Keep per-event ranks, user ID, step, dataset, model and seed.

In text/figures distinguish **full filtered-catalog evaluation** from full Amazon-catalog evaluation. Unreviewed items are unknown, not proven negatives. Data do not contain randomized recommendation impressions, so offline ranking is **not** online treatment lift.

## Fair comparisons

Two-tower recomputes user embedding with each new rating; frozen *weights* do not mean frozen user representation. Compare separate trained static extra-capacity, `d` vector TTT, **`d²`-feature capacity-matched vector TTT**, and matrix TTT. Use consistent history/candidates/seeds/tuning budgets. Report per-user bootstrap uncertainty and history-bin counts. Synthetic preference switches must be labeled as such.
