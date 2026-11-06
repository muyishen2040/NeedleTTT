# Experiment and benchmarking entrypoints

Samuel Shen: place reproducible dataset preparation, training and rolling-evaluation CLI scripts here, with config files in `configs/`. Maggie Pai: add Needle/CUDA benchmark runners here, saving results under gitignored `artifacts/`. No real data or trained experiments are bundled in the bootstrap.

Suggested future commands (not implemented yet):
- `python scripts/prepare_dataset.py --config configs/movielens.yaml`
- `python scripts/train.py --config configs/amazon_video_games.yaml`
- `python scripts/evaluate.py --checkpoint ...`
- `python scripts/benchmark_fast_weights.py --device cuda`

Please do not claim a command exists until its associated Linear ticket has added and tested it.
