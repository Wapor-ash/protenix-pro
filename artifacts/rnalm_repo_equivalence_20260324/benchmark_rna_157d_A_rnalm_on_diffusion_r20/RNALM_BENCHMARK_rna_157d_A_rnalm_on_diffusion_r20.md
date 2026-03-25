# RNALM Fixed-Case Benchmark

- Date: 2026-03-24
- Sample: `rna_157d_A`
- RNALM mode: `rnalm_on_diffusion`
- Data root: `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2`
- GPU: NVIDIA H800
- Warmup runs: 5
- Timed runs: 20

## Timing Table

| Repo | dataset_item_load_s | pairformer_s | forward_mean_s | forward_median_s | forward_std_s | forward_p05_s | forward_p95_s | forward_min_s | forward_max_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| protenix_new | 0.503753 | 0.720330 | 0.171367 | 0.164272 | 0.052336 | 0.097979 | 0.257151 | 0.082997 | 0.274961 |
| protenix_pro | 0.367663 | 0.699595 | 0.161705 | 0.152871 | 0.064567 | 0.078095 | 0.277638 | 0.061459 | 0.333424 |

## Repo Delta

- forward median delta (`protenix_pro - protenix_new`): -0.011401 s
- forward median delta percent vs `protenix_new`: -6.94%

## Numerical Check

- raw data equal: True
- prepared model input equal: True
- pairformer exact same: False
- pairformer max abs diff: 0.0003662109375
- prediction exact same: False
- prediction max abs diff: 0.000732421875

## Note

- This benchmark fixes one case and repeats the same GPU forward path many times to reduce one-shot timing noise.