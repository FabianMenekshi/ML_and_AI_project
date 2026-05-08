|   bits | granularity   | symmetry   | calibration        |   n_seeds | final_val_bpb    |   delta_bpb |   size_MB |
|-------:|:--------------|:-----------|:-------------------|----------:|:-----------------|------------:|----------:|
|      4 | per-channel   | asymmetric | percentile-clipped |         3 | 4.2878 ± 0.3011  |      2.9812 |      8.35 |
|      4 | per-channel   | asymmetric | uncalibrated       |         3 | 4.5090 ± 0.5575  |      3.2027 |      8.35 |
|      4 | per-channel   | symmetric  | percentile-clipped |         3 | 2.6133 ± 0.0333  |      1.3068 |      7.82 |
|      4 | per-channel   | symmetric  | uncalibrated       |         3 | 2.5749 ± 0.0328  |      1.2683 |      7.82 |
|      4 | per-tensor    | asymmetric | percentile-clipped |         3 | 8.0454 ± 1.8683  |      6.739  |      6.26 |
|      4 | per-tensor    | asymmetric | uncalibrated       |         3 | 6.9151 ± 1.2805  |      5.6084 |      6.19 |
|      4 | per-tensor    | symmetric  | percentile-clipped |         3 | 6.5765 ± 0.3778  |      5.2701 |      6    |
|      4 | per-tensor    | symmetric  | uncalibrated       |         3 | 6.6045 ± 0.3734  |      5.2981 |      5.78 |
|      8 | per-channel   | asymmetric | percentile-clipped |         3 | 15.3462 ± 9.7565 |     14.0397 |     16.05 |
|      8 | per-channel   | asymmetric | uncalibrated       |         3 | 7.5272 ± 3.1482  |      6.2211 |     16.05 |
|      8 | per-channel   | symmetric  | percentile-clipped |         3 | 1.3105 ± 0.0012  |      0.004  |     15.75 |
|      8 | per-channel   | symmetric  | uncalibrated       |         3 | 1.3101 ± 0.0011  |      0.0038 |     15.75 |
|      8 | per-tensor    | asymmetric | percentile-clipped |         3 | 4.1052 ± 0.0000  |      2.7988 |     14.17 |
|      8 | per-tensor    | asymmetric | uncalibrated       |         3 | 4.1052 ± 0.0000  |      2.7986 |     14.07 |
|      8 | per-tensor    | symmetric  | percentile-clipped |         3 | 1.3262 ± 0.0055  |      0.0197 |     14.02 |
|      8 | per-tensor    | symmetric  | uncalibrated       |         3 | 1.3293 ± 0.0032  |      0.0229 |     13.8  |