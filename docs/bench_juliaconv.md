# Julia Convention Refactor Benchmarks

This file tracks per-layer timing comparisons between the existing feature-last layers and the new Julia-convention (feature-first, batch-last) duplicates.

## How to Run

Example:

```bash
CASE=esmfold_attention julia --project=. scripts/bench_juliaconv.jl
```

Available cases are listed by running the script with an unknown CASE.

## Results (CPU)

| Layer | Variant | Shape | Min | Median | Memory | Allocs | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ESMFoldAttention | feature-last | B=1, L=256, C=1024 | 28.336 ms | 35.001 ms | 59.034 MiB | 103 | BenchmarkTools, `--compiled-modules=no` |
| ESMFoldAttentionJL | feature-first | C=1024, L=256, B=1 | 26.423 ms | 32.569 ms | 67.035 MiB | 87 | BenchmarkTools, `--compiled-modules=no` |
| SequenceToPair | feature-last | B=1, L=256, C_s=1024, C_z=128 | 18.751 ms | 24.160 ms | 99.130 MiB | 69 | BenchmarkTools, `--compiled-modules=no` |
| SequenceToPairJL | feature-first | C_s=1024, L=256, B=1 | 55.842 ms | 63.707 ms | 131.255 MiB | 66 | BenchmarkTools, `--compiled-modules=no` |
| PairToSequence | feature-last | B=1, L=256, C_z=128 | 59.788 ms | 67.644 ms | 104.502 MiB | 34 | BenchmarkTools, `--compiled-modules=no` |
| PairToSequenceJL | feature-first | C_z=128, L=256, B=1 | 29.378 ms | 33.396 ms | 104.502 MiB | 34 | BenchmarkTools, `--compiled-modules=no` |
| TriangleAttention (start) | feature-last | B=1, L=128, C_z=128 | 267.745 ms | 346.628 ms | 320.947 MiB | 273 | BenchmarkTools, `--compiled-modules=no` |
| TriangleAttentionJL (start) | feature-first | C_z=128, L=128, B=1 | 213.423 ms | 269.667 ms | 321.070 MiB | 223 | BenchmarkTools, `--compiled-modules=no` |
| TriangleAttention (end) | feature-last | B=1, L=128, C_z=128 | 338.452 ms | 434.267 ms | 337.011 MiB | 318 | BenchmarkTools, `--compiled-modules=no` |
| TriangleAttentionJL (end) | feature-first | C_z=128, L=128, B=1 | 314.554 ms | 386.813 ms | 305.007 MiB | 214 | BenchmarkTools, `--compiled-modules=no` |
| TriangleMultiplicationOutgoing | feature-last | B=1, L=128, C_z=128 | 113.966 ms | 129.924 ms | 184.256 MiB | 162 | BenchmarkTools, `--compiled-modules=no` |
| TriangleMultiplicationOutgoingJL | feature-first | C_z=128, L=128, B=1 | 86.228 ms | 93.999 ms | 216.255 MiB | 138 | BenchmarkTools, `--compiled-modules=no` |
| TriangleMultiplicationIncoming | feature-last | B=1, L=128, C_z=128 | 111.457 ms | 123.931 ms | 184.256 MiB | 162 | BenchmarkTools, `--compiled-modules=no` |
| TriangleMultiplicationIncomingJL | feature-first | C_z=128, L=128, B=1 | 86.942 ms | 90.589 ms | 216.255 MiB | 138 | BenchmarkTools, `--compiled-modules=no` |
| TriangularSelfAttentionBlock | feature-last | B=1, L=128, C_s=1024, C_z=128 | 1.072 s | 1.072 s | 1.225 GiB | 1290 | BenchmarkTools, `--compiled-modules=no` |
| TriangularSelfAttentionBlockJL | feature-first | C_s=1024, L=128, B=1 | 875.180 ms | 913.448 ms | 1.308 GiB | 1091 | BenchmarkTools, `--compiled-modules=no` |
| StructureModule | feature-last | B=1, L=128, C_s=384, C_z=128 | 1.016 s | 1.016 s | 1.194 GiB | 36409 | BenchmarkTools, `--compiled-modules=no` |
| StructureModuleJL | feature-first | C_s=384, L=128, B=1 | 1.055 s | 1.055 s | 1.204 GiB | 36459 | BenchmarkTools, `--compiled-modules=no` |
| StructureModuleJLCore | feature-first | C_s=384, L=128, B=1 | 1.006 s | 1.006 s | 1.245 GiB | 30960 |  |
| FoldingTrunk | feature-last | B=1, L=64, no_recycles=0, num_blocks=4 | 1.114 s | 1.114 s | 1.471 GiB | 41687 | BenchmarkTools, `--compiled-modules=no` |
| FoldingTrunkJL | feature-first | L=64, B=1, no_recycles=0, num_blocks=4 | 1.772 s | 1.772 s | 1.558 GiB | 40910 | BenchmarkTools, `--compiled-modules=no` |
| FoldingTrunkJLCore | feature-first | L=64, B=1, no_recycles=0, num_blocks=4 | 877.005 ms | 881.891 ms | 1.573 GiB | 35023 |  |
