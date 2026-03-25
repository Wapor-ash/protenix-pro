# RNALM Repo Equivalence GPU Report

- Date: 2026-03-24
- GPU: NVIDIA H800
- Python: /inspire/ssd/project/sais-bio/public/ash_proj/conda/envs/tune_protenix/bin/python
- Data root: `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2`
- Model checkpoint: `protenix_base_20250630_v1.0.0.pt`
- Execution mode: GPU, fp32, `triangle_attention=torch`, `triangle_multiplicative=torch`, `N_cycle=1`, `N_step=2`, `N_sample=1`, crop disabled
- Minimal-instance block counts: template=0, msa=1, pairformer=2, diffusion(atom/transformer/decoder)=1/2/1, confidence=1

## Verdict

### rna_157d_A / rnalm_off
- raw data equal: True
- prepared model input equal: True
- pairformer tensors equal: False
- final prediction equal: False
- checkpoint missing keys equal: True
- basic metadata equal: True
- protenix_new forward_avg_s: 0.426626
- protenix_pro forward_avg_s: 0.150919

### rna_157d_A / rnalm_on_diffusion
- raw data equal: True
- prepared model input equal: True
- pairformer tensors equal: False
- final prediction equal: False
- checkpoint missing keys equal: True
- basic metadata equal: True
- protenix_new forward_avg_s: 0.420773
- protenix_pro forward_avg_s: 0.084681

### hybrid_165d_A / rnalm_off
- raw data equal: True
- prepared model input equal: True
- pairformer tensors equal: False
- final prediction equal: False
- checkpoint missing keys equal: True
- basic metadata equal: True
- protenix_new forward_avg_s: 0.666963
- protenix_pro forward_avg_s: 0.073197

### hybrid_165d_A / rnalm_on_diffusion
- raw data equal: True
- prepared model input equal: True
- pairformer tensors equal: False
- final prediction equal: False
- checkpoint missing keys equal: True
- basic metadata equal: True
- protenix_new forward_avg_s: 0.158438
- protenix_pro forward_avg_s: 0.595595

## RNALM On/Off Inside Each Repo

### rna_157d_A
- protenix_new raw input exact same: False
- protenix_new pairformer exact same: False
- protenix_new prediction exact same: False
- protenix_new off missing keys: []
- protenix_new on missing keys: ['rnalm_projection.weight']
- protenix_pro raw input exact same: False
- protenix_pro pairformer exact same: False
- protenix_pro prediction exact same: False
- protenix_pro off missing keys: []
- protenix_pro on missing keys: ['rnalm_projection.weight']

### hybrid_165d_A
- protenix_new raw input exact same: False
- protenix_new pairformer exact same: False
- protenix_new prediction exact same: False
- protenix_new off missing keys: []
- protenix_new on missing keys: ['rnalm_projection.weight']
- protenix_pro raw input exact same: False
- protenix_pro pairformer exact same: False
- protenix_pro prediction exact same: False
- protenix_pro off missing keys: []
- protenix_pro on missing keys: ['rnalm_projection.weight']

## Notes

- If `rnalm_on_diffusion` reports extra missing keys, those are the RNALM projection parameters absent from the base checkpoint and left at their zero initialization.
- Performance numbers are minimal-instance timings, useful for relative comparison only.
