# DRL for GCS - concise guide

## Files
- cluster_permeability_mds_kmeans.ipynb
- build_robust_feasibility_mask.ipynb
- co2-ppo-cma-human.ipynb
- test_env_run.ipynb
- plot_train_eval_rewards.ipynb
- plot_pressure_penalty.ipynb
- co2storage_env_gru.py
- cnn_encoder_co2storage.py
- co2storage_ppo_utils.py
- cma_es_baseline_gcs.py
- train_ppo_gcs_multiseed.py
- surrogate_model_gru.py
- fno_3d_LKA_test.py
- h5_utils.py

## Data and weights
- train_data.h5
- valid_data.h5
- BulkVolume.h5
- robust_feasible_map_3d_gaussian_smoothed.npy  (created by build_robust_feasibility_mask.ipynb)
- weights/epoch_49_820_15_5_sat.pth
- weights/epoch_49_820_15_5_pres.pth
- normalizers/820_c5_pressure_15_5_train_aug_output.pkl

## Run order
1) cluster_permeability_mds_kmeans.ipynb
2) build_robust_feasibility_mask.ipynb
3) test_env_run.ipynb
4) Train PPO
   python train_ppo_gcs_multiseed.py
5) Plot learning curves
   plot_train_eval_rewards.ipynb
6) CMA-ES baseline
   python cma_es_baseline_gcs.py

## Notes
- Env file: co2storage_env_gru.py
- PPO utils: co2storage_ppo_utils.py
- CNN encoder: cnn_encoder_co2storage.py
- Surrogate wrapper: surrogate_model_gru.py
- Overview notebook: co2-ppo-cma-human.ipynb
- Python 3.10 recommended. Install PyTorch and Stable Baselines3.
