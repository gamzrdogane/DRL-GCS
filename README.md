# co2-ppo-cma-human — DRL for GCS

## Purpose
Learn injection policies for GCS with PPO. Compare to CMA-ES. Use a GRU or FNO surrogate.

## Quick start
1) Run `cluster_permeability_mds_kmeans.ipynb`. Select representative scenarios.  
2) Run `build_robust_feasibility_mask.ipynb`. Save `robust_feasible_map_3d_gaussian_smoothed.npy`.  
3) Run `test_env_run.ipynb`. Check shapes and rewards.  
4) Train PPO.  
   ```bash
   python train_ppo_gcs_multiseed.py
   ```
5) Plot curves.  
   Open `plot_train_eval_rewards.ipynb`.  
6) Run CMA-ES baseline.  
   ```bash
   python cma_es_baseline_gcs.py
   ```

## Files and roles
- `cluster_permeability_mds_kmeans.ipynb`: Clusters permeability fields. Picks medoids. Plots.  
- `build_robust_feasibility_mask.ipynb`: Builds the feasibility mask. Screens boundary leakage. Saves NPY.  
- `co2-ppo-cma-human.ipynb`: Overview of PPO and CMA-ES workflow.  
- `plot_train_eval_rewards.ipynb`: Plots training and evaluation curves from logs.  
- `test_env_run.ipynb`: Smoke test for env and surrogate. Runs a short rollout.  
- `cma_es_baseline_gcs.py`: Runs CMA-ES baseline and logs best genomes.  
- `cnn_encoder_co2storage.py`: CNN feature extractor for PPO.  
- `co2storage_env_gru.py`: GCS environment with three wells. 
- `co2storage_ppo_utils.py`: Eval callback and plotting helpers. 
- `surrogate_model_gru.py`: Loads GRU or FNO surrogate weights. Provides `.inference(...)`.  
- `fno_3d_LKA_test.py`: FNO and local attention blocks used by the surrogate.  
- `h5_utils.py`: HDF5 helpers for quick data checks.  
- `plot_pressure_penalty.py`: Plots the logistic pressure penalty curve.  
- `robust_feasible_map_3d_gaussian_smoothed.npy`: Feasibility mask file created by the mask notebook.  
- `Data/`: Optional folder for H5 files and weights if you do not keep them in the root.  

## Expected data and weights
- `train_data.h5`, `valid_data.h5`, `BulkVolume.h5`.  
- `weights/epoch_49_820_15_5_sat.pth`, `weights/epoch_49_820_15_5_pres.pth`.  
- `normalizers/820_c5_pressure_15_5_train_aug_output.pkl`.  

## Environment
- Python 3.10 recommended.  
- Install PyTorch and Stable Baselines3.  
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install stable-baselines3==2.3.2 gym==0.26.2 gym-notices h5py numpy pandas matplotlib cma tqdm
  ```
- Add repo to `PYTHONPATH` before running.  

