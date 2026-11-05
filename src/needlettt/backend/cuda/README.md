# Planned CUDA integration (DEE-15–DEE-18)

No GPU implementation exists in the bootstrap.

Goal: fuse the sequential matrix update `W <- W - eta*outer(W@k-v,k)/(k@k+eps)` for each independent user. Benchmark batch size, number of events, fast-state dimension, kernel launches and latency. Match the Needle reference numerically before claiming speedup. A custom backward kernel is optional.

Keep course Needle source in a **separate, private** checkout until licensing and class sharing rules are confirmed. Agree on the build/import boundary in DEE-5 before adding CUDA source.
