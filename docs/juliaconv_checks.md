# Julia-Convention Checks

Heuristic GPU sniff: static scan for loops, @inbounds, or setindex! in method source; no GPU execution.

| Layer | Parity Max Diff | Zygote Input | Zygote Params | GPU Sniff | Base Median | JL Median | Ratio | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LinearFirst | 0.0 | ok | ok | no obvious scalar indexing | 262.638 ns | 342.635 ns | 1.305 |  |
| LayerNormFirst | 2.3841858e-7 | ok | ok | no obvious scalar indexing | 1.575 μs | 707.523 ns | 0.449 |  |
| ESMFoldAttentionJL | 0.0 | ok | ok | no obvious scalar indexing | 41.917 μs | 40.750 μs | 0.972 |  |
| SequenceToPairJL | 1.9073486e-6 | ok | ok | no obvious scalar indexing | 25.833 μs | 48.542 μs | 1.879 |  |
| PairToSequenceJL | 9.536743e-7 | ok | ok | no obvious scalar indexing | 42.042 μs | 29.292 μs | 0.697 |  |
| ResidueMLPJL | 7.1525574e-7 | ok | ok | no obvious scalar indexing | 21.750 μs | 24.708 μs | 1.136 |  |
| OFMultiheadAttentionJL | 0.0 | ok | ok | no obvious scalar indexing | 27.625 μs | 23.125 μs | 0.837 |  |
| TriangleAttentionJL(start) | 4.7683716e-7 | ok | ok | no obvious scalar indexing | 463.667 μs | 575.250 μs | 1.241 |  |
| TriangleAttentionJL(end) | 5.066395e-7 | ok | ok | no obvious scalar indexing | 465.292 μs | 458.625 μs | 0.986 |  |
| TriangleMultiplicativeUpdateJL(out) | 1.4901161e-6 | ok | ok | no obvious scalar indexing | 199.958 μs | 178.792 μs | 0.894 |  |
| TriangleMultiplicativeUpdateJL(in) | 1.3783574e-6 | ok | ok | no obvious scalar indexing | 210.708 μs | 183.792 μs | 0.872 |  |
| TriangularSelfAttentionBlockJL | 4.7683716e-6 | ok | ok | no obvious scalar indexing | 1.086 ms | 920.833 μs | 0.848 |  |
| RelativePositionJL | 0.0 | ok | ok | no obvious scalar indexing | 10.681 μs | 4.923 μs | 0.461 | input integer; skipped |
| PointProjectionJL | 0.0 | ok | ok | no obvious scalar indexing | 21.791 μs | 20.958 μs | 0.962 |  |
| InvariantPointAttentionJL | 0.0 | ok | ok | no obvious scalar indexing | 138.125 μs | 180.500 μs | 1.307 |  |
| BackboneUpdateJL | 0.0 | ok | ok | no obvious scalar indexing | 221.435 ns | 435.930 ns | 1.969 |  |
| StructureModuleTransitionJL | 4.7683716e-7 | ok | ok | no obvious scalar indexing | 7.972 μs | 6.528 μs | 0.819 |  |
| AngleResnetJL | 0.0 | ok | ok | no obvious scalar indexing | 4.213 μs | 5.964 μs | 1.416 |  |
| StructureModuleJL | 0.0 | ok | ok | no obvious scalar indexing | 603.958 μs | 294.750 μs | 0.488 | skipped |
| StructureModuleJLCore | 0.0 | ok | ok | no obvious scalar indexing | 472.166 μs | 317.584 μs | 0.673 | skipped |
| FoldingTrunkJL | 0.0 | ok | ok | no obvious scalar indexing | 374.875 μs | 467.750 μs | 1.248 | skipped |
| FoldingTrunkJLCore | 0.0 | ok | ok | no obvious scalar indexing | 339.917 μs | 479.375 μs | 1.41 | skipped |
| ESMFoldModelJL | 0.0 | ok | ok | no obvious scalar indexing | 1.183 ms | 680.459 μs | 0.575 | skipped |
