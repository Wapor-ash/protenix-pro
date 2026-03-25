# RNALM Fixed-Case 20x Benchmark Summary

- Date: 2026-03-24
- GPU: NVIDIA H800
- Sample: `rna_157d_A`
- Data root: `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2`
- Warmup: `5`
- Timed runs: `20`
- Script: `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix/rnalm_repo_equivalence_gpu_check_20260324.py`

## Forward Timing Summary

| Mode | Repo | forward_mean_s | forward_median_s | forward_std_s | forward_p05_s | forward_p95_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `rnalm_off` | `protenix_new` | 0.153863 | 0.134063 | 0.087911 | 0.069051 | 0.249046 |
| `rnalm_off` | `protenix_pro` | 0.156723 | 0.163537 | 0.053696 | 0.064582 | 0.221589 |
| `rnalm_on_diffusion` | `protenix_new` | 0.171367 | 0.164272 | 0.052336 | 0.097979 | 0.257151 |
| `rnalm_on_diffusion` | `protenix_pro` | 0.161705 | 0.152871 | 0.064567 | 0.078095 | 0.277638 |

## Cross-Repo Median Delta

| Mode | `protenix_pro - protenix_new` |
| --- | ---: |
| `rnalm_off` | +0.029475 s (`+21.99%`) |
| `rnalm_on_diffusion` | -0.011401 s (`-6.94%`) |

## Numerical Consistency

| Mode | raw equal | prepared equal | pairformer max abs diff | prediction max abs diff |
| --- | --- | --- | ---: | ---: |
| `rnalm_off` | `True` | `True` | 0.0003662109375 | 0.0009765625 |
| `rnalm_on_diffusion` | `True` | `True` | 0.0003662109375 | 0.000732421875 |

## Reading

- The fixed-case 20x benchmark does not show a stable repo ordering.
- With `rnalm_off`, `protenix_pro` median forward is slower by about `22%`.
- With `rnalm_on_diffusion`, `protenix_pro` median forward is faster by about `7%`.
- Because the direction flips between `off` and `on`, the current evidence does not support a strong claim that one repo is systematically faster on this minimal instance.
- The numerical equivalence conclusion remains unchanged: raw input and prepared model input are identical across repos, and forward drift remains very small.
