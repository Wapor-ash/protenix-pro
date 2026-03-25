# RNALM Fixed-Case Benchmark

- Date: 2026-03-24
- Sample: `rna_157d_A`
- RNALM mode: `rnalm_off`
- Data root: `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2`
- GPU: NVIDIA H800
- Warmup runs: 5
- Timed runs: 20

## Timing Table

| Repo | dataset_item_load_s | pairformer_s | forward_mean_s | forward_median_s | forward_std_s | forward_p05_s | forward_p95_s | forward_min_s | forward_max_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| protenix_new | 0.550838 | 1.265381 | 0.153863 | 0.134063 | 0.087911 | 0.069051 | 0.249046 | 0.060968 | 0.471435 |
| protenix_pro | 0.674199 | 1.148666 | 0.156723 | 0.163537 | 0.053696 | 0.064582 | 0.221589 | 0.060103 | 0.240022 |

## Repo Delta

- forward median delta (`protenix_pro - protenix_new`): 0.029475 s
- forward median delta percent vs `protenix_new`: 21.99%

## Numerical Check

- raw data equal: True
- prepared model input equal: True
- pairformer exact same: False
- pairformer max abs diff: 0.0003662109375
- prediction exact same: False
- prediction max abs diff: 0.0009765625

## Note

- This benchmark fixes one case and repeats the same GPU forward path many times to reduce one-shot timing noise.