| method       |   bits | granularity   | symmetry   | calibration        |   n_seeds | final_val_bpb   |   delta_bpb |   size_MB |
|:-------------|-------:|:--------------|:-----------|:-------------------|----------:|:----------------|------------:|----------:|
| Uniform INT4 |      4 | per-channel   | symmetric  | uncalibrated       |         3 | 2.5749 ± 0.0328 |      1.2683 |      7.82 |
| NF4          |      4 | per-tensor    | codebook   | uncalibrated       |         3 | 7.5238 ± 2.5315 |      6.2175 |      6.47 |
| NF4          |      4 | per-channel   | codebook   | uncalibrated       |         3 | 1.8163 ± 0.0222 |      0.51   |      7.91 |
| NF4          |      4 | per-channel   | codebook   | percentile-clipped |         3 | 1.8211 ± 0.0208 |      0.5148 |      7.91 |