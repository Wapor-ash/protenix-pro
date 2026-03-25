# RNALM Multisample GPU Matrix Report

- Samples: 1idv_A, 157d_A, 2m1v_A, 2m1v_B, 377d_A, 165d_A
- Modes: rnalm_off, rnalm_on_diffusion
- Warmup: 2, repeats: 7

## Cross-Repo Exactness

- raw_input_feature_dict: 12/12 exact, all_exact=True, max_abs_diff=0.0
- prepared_input_feature_dict: 12/12 exact, all_exact=True, max_abs_diff=0.0
- label_dict: 12/12 exact, all_exact=True, max_abs_diff=0.0
- label_full_dict: 12/12 exact, all_exact=True, max_abs_diff=0.0
- pairformer: 3/12 exact, all_exact=False, max_abs_diff=0.0008544921875
- prediction: 0/12 exact, all_exact=False, max_abs_diff=0.005118370056152344

## Cross-Repo Performance Stats (`protenix_pro - protenix_new`)

- dataset_init_delta_s: mean=-0.008496, 95% CI=[-0.029990, 0.012806], median=-0.004661, sign_test_p=1.000000, n=12
- dataset_item_load_delta_s: mean=0.019809, 95% CI=[-0.082812, 0.131059], median=-0.000814, sign_test_p=1.000000, n=12
- model_init_delta_s: mean=-0.419640, 95% CI=[-2.416265, 1.529703], median=-0.097052, sign_test_p=1.000000, n=12
- checkpoint_load_delta_s: mean=-0.022938, 95% CI=[-0.059695, 0.003264], median=0.001048, sign_test_p=0.774414, n=12
- pairformer_delta_s: mean=-0.046679, 95% CI=[-0.168463, 0.045726], median=0.010361, sign_test_p=0.774414, n=12
- forward_mean_delta_s: mean=-0.013113, 95% CI=[-0.020718, -0.004805], median=-0.012144, sign_test_p=0.006348, n=12
- forward_median_delta_s: mean=-0.013495, 95% CI=[-0.026116, 0.000223], median=-0.016856, sign_test_p=0.145996, n=12
- forward_mean_delta_pct: mean=-7.426613, 95% CI=[-12.398166, -1.637182], median=-6.997948, sign_test_p=0.006348, n=12
- forward_median_delta_pct: mean=-5.942218, 95% CI=[-14.089341, 3.418002], median=-10.274163, sign_test_p=0.145996, n=12

## Within-Repo On/Off Exactness

- protenix_new:raw_input_feature_dict: 0/6 exact, all_exact=False
- protenix_new:prepared_input_feature_dict: 0/6 exact, all_exact=False
- protenix_new:pairformer: 0/6 exact, all_exact=False
- protenix_new:prediction: 0/6 exact, all_exact=False
- protenix_pro:raw_input_feature_dict: 0/6 exact, all_exact=False
- protenix_pro:prepared_input_feature_dict: 0/6 exact, all_exact=False
- protenix_pro:pairformer: 0/6 exact, all_exact=False
- protenix_pro:prediction: 0/6 exact, all_exact=False

## Within-Repo On/Off Performance Deltas (`on - off`)

- protenix_new:forward_mean_delta_s: mean=-0.001956, 95% CI=[-0.024650, 0.020334], median=0.000960, sign_test_p=1.000000, n=6
- protenix_new:forward_median_delta_s: mean=-0.004472, 95% CI=[-0.040051, 0.024570], median=0.009413, sign_test_p=1.000000, n=6
- protenix_new:dataset_item_load_delta_s: mean=0.111306, 95% CI=[0.018167, 0.220357], median=0.083826, sign_test_p=0.218750, n=6
- protenix_new:pairformer_delta_s: mean=-0.080996, 95% CI=[-0.334518, 0.140268], median=-0.044818, sign_test_p=0.687500, n=6
- protenix_pro:forward_mean_delta_s: mean=0.001567, 95% CI=[-0.010409, 0.015549], median=-0.003889, sign_test_p=1.000000, n=6
- protenix_pro:forward_median_delta_s: mean=-0.006809, 95% CI=[-0.026315, 0.010672], median=-0.003091, sign_test_p=1.000000, n=6
- protenix_pro:dataset_item_load_delta_s: mean=0.071743, 95% CI=[-0.053201, 0.193302], median=0.132636, sign_test_p=0.687500, n=6
- protenix_pro:pairformer_delta_s: mean=0.094080, 95% CI=[0.003068, 0.203072], median=0.078012, sign_test_p=0.687500, n=6
