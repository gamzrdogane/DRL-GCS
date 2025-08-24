###############################################################################
# PPO Training for GCS Optimization with 5 Seeds and CNN Policy
#
# This script trains a PPO agent using a CNN-based policy to optimize 
# injection strategies in GCS (Geological CO₂ Storage) environments.
# It uses 5 distinct random seeds and 8 parallel environments per run,
# consistent with the setup and hyperparameter configuration described 
# in the manuscript. Total training steps are fixed at 100,000 per seed.
###############################################################################

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
import torch
import gc
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from surrogate_model_gru import SurrogateModel
from co2storage_env_gru import CO2StorageEnv

def make_single_env(data_path, seed=0, device='cuda'):
    def _init():
        env = CO2StorageEnv(
            max_time=20,
            data_path=data_path,
            bulk_data_path='BulkVolume.h5',
            feasible_map_path="robust_feasible_map_3d_gaussian_smoothed.npy",
            device=torch.device(device)
        )
        env.seed(seed)
        return env
    return DummyVecEnv([_init])

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def make_train_env_fn(rank, seed=0, device='cuda'):
    def _init():
        env = CO2StorageEnv(
            max_time=20,
            data_path="train_data.h5",
            bulk_data_path='BulkVolume.h5',
            feasible_map_path="robust_feasible_map_3d_gaussian_smoothed.npy",
            device=torch.device(device)
        )
        env.seed(seed + rank)
        return env
    return _init

if __name__ == "__main__":

    num_cpu = 8
    seed_list = [1, 2, 3, 4, 5]

    for seed in seed_list:
        print("==========================================")
        print(f"  STARTING RUN WITH SEED = {seed}")
        print("==========================================")

        start_time = time.time()

        train_log_dir = f"./train_logs/seed_{seed}"
        eval_log_dir  = f"./eval_logs/seed_{seed}"
        best_model_dir = f"./best_model_eval/seed_{seed}"

        os.makedirs(train_log_dir, exist_ok=True)
        os.makedirs(eval_log_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)

        env_fns = [make_train_env_fn(i, seed=seed, device='cuda') for i in range(num_cpu)]
        vec_train = SubprocVecEnv(env_fns))
        vec_train = VecMonitor(vec_train, filename=os.path.join(train_log_dir, "monitor.csv"))

        train_eval_env = make_single_env("train_data.h5", seed=seed+50, device='cuda')
        vec_eval = make_single_env("valid_data.h5", seed=seed+100, device='cuda')
        vec_eval = VecMonitor(vec_eval, filename=os.path.join(eval_log_dir, "monitor.csv"))

        policy_kwargs = dict(
            features_extractor_class=LargerCNN,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            log_std_init=-1.5
        )

        model = PPO(
            policy="CnnPolicy",
            env=vec_train,
            learning_rate=1e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=20,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.0005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(".", "tensorboard_log", f"seed_{seed}"),
            device="auto",
            verbose=1,
            seed=seed
        )

        train_callback = CustomEvalCallback(
            eval_env=train_eval_env,
            n_eval_episodes=1,
            eval_freq=100,
            log_path=os.path.join(train_log_dir, "results_train"),
            best_model_save_path=None,
            deterministic=True,
            verbose=1
        )
        eval_callback = CustomEvalCallback(
            eval_env=vec_eval,
            n_eval_episodes=1,
            eval_freq=100,
            log_path=os.path.join(eval_log_dir, "results_eval"),
            best_model_save_path=best_model_dir,
            deterministic=True,
            verbose=1
        )
        callback_list = CallbackList([train_callback, eval_callback])

        total_timesteps = 100_000
        model.learn(total_timesteps=total_timesteps, callback=callback_list)

        final_model_path = f"./trained_models/ppo_co2storage_final_seed_{seed}"
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        model.save(final_model_path)

        print(f"Training complete for seed = {seed}")

        fig = plot_learning(train_log_dir, case='train', window=20)
        if fig is not None:
            plt.savefig(f"train_curve_seed_{seed}.png")
            plt.close(fig)

        fig = plot_learning(eval_log_dir, case='eval', window=1)
        if fig is not None:
            plt.savefig(f"eval_curve_seed_{seed}.png")
            plt.close(fig)

        print(f"Run with seed={seed} is DONE.\n")

        end_time = time.time()
        duration = end_time - start_time
        print(f"Time taken for seed {seed}: {duration:.2f} seconds")

        clear_memory()
