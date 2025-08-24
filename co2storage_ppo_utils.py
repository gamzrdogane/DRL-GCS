"""
Utilities used across PPO experiments for the CO2 storage study.

Includes
- Imports shared by later cells (env, CNN, training scripts)
- moving_average for smoothing curves
- custom_evaluate_policy for single-env VecEnv evaluation
- CustomEvalCallback for periodic eval, best-model checkpointing, and NPZ logging
- plot_learning to build rolling-mean reward curves from SB3 Monitor CSVs

Corresponds to the manuscript's RL training and evaluation setup, and supports
the learning-curve plots reported in the Results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

import gym
from gym import spaces
import h5py

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback  # CallbackList used later
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.utils import set_random_seed

from surrogate_model_gru import SurrogateModel


def moving_average(values: np.ndarray, window: int = 50) -> np.ndarray:
    """
    Compute a simple moving average over a 1D array.
    """
    if window <= 1:
        return np.asarray(values, dtype=float)
    weights = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, weights, mode="valid")


def custom_evaluate_policy(
    model: PPO,
    eval_env: DummyVecEnv,
    n_eval_episodes: int = 5,
    deterministic: bool = True
) -> tuple[float, float]:
    """
    Evaluate the model on n_eval_episodes using eval_env.
    Returns (mean_reward, std_reward).

    IMPORTANT: eval_env must wrap exactly one environment so 'done' is a single bool.
    """
    episode_rewards: list[float] = []

    for _ in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = eval_env.step(action)
            total_reward += float(reward[0])  # ensure Python float
            done = bool(done[0])              # ensure Python bool

        episode_rewards.append(total_reward)

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
    return mean_reward, std_reward


class CustomEvalCallback(BaseCallback):
    """
    Periodically evaluate the current policy on a held-out environment and
    optionally save the best model checkpoint.

    Notes
    -----
    - Assumes eval_env has exactly one environment.
    - Saves both mean and std rewards into an NPZ file for plotting later.
    """
    def __init__(
        self,
        eval_env: DummyVecEnv,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = "./eval_logs/results_eval",
        best_model_save_path: str | None = "./best_model_eval",
        deterministic: bool = True,
        verbose: int = 1
    ):
        super().__init__(verbose)

        self.eval_env = eval_env
        self.n_eval_episodes = int(n_eval_episodes)
        self.eval_freq = int(eval_freq)
        self.deterministic = deterministic

        self.best_mean_reward = -np.inf
        self.best_model_save_path = best_model_save_path
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

        self.log_path = log_path
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        self.evaluations_timesteps: list[int] = []
        self.evaluations_mean: list[float] = []
        self.evaluations_std: list[float] = []
        self.epochs: list[int] = []
        self.epoch_rewards: list[float] = []

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.num_timesteps % self.eval_freq == 0):
            mean_reward, std_reward = custom_evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic
            )

            epoch = int(self.num_timesteps // self.eval_freq)
            self.epochs.append(epoch)
            self.epoch_rewards.append(float(mean_reward))

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, epoch={epoch}, "
                    f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )

            # log buffers
            self.evaluations_timesteps.append(int(self.num_timesteps))
            self.evaluations_mean.append(float(mean_reward))
            self.evaluations_std.append(float(std_reward))

            # best model checkpoint
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    if self.verbose > 0:
                        print("Saving model to:", self.best_model_save_path)
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))

            # persist results as .npz for later plotting
            if self.log_path is not None:
                np.savez(
                    self.log_path,
                    timesteps=np.asarray(self.evaluations_timesteps, dtype=np.int64),
                    mean=np.asarray(self.evaluations_mean, dtype=np.float32),
                    std=np.asarray(self.evaluations_std, dtype=np.float32),
                    epochs=np.asarray(self.epochs, dtype=np.int64),
                    epoch_rewards=np.asarray(self.epoch_rewards, dtype=np.float32),
                )

        return True


# Unified Plotting Function
import pandas as pd

def plot_learning(log_dir: str, case: str = "train", window: int = 20):
    """
    Generate a rolling-mean reward curve from Stable-Baselines3 Monitor files.

    Parameters
    ----------
    log_dir : str
        Directory containing the Monitor CSV files (one per environment).
    case : str
        A label appended to the plot title and the output figure.
    window : int
        Rolling-mean window size in episodes.

    Returns
    -------
    matplotlib.figure.Figure
        Figure handle for further processing or saving to disk.
    """
    # 1) Locate Monitor files
    monitor_files = [
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if f.endswith(".csv") and "monitor" in f
    ]
    if not monitor_files:
        raise FileNotFoundError(
            f"No Stable-Baselines3 Monitor CSV files found in: {log_dir}"
        )

    # 2) Aggregate episode rewards across all environments
    df_list = []
    for file in monitor_files:
        # The first line is a comment with meta information
        df = pd.read_csv(file, skiprows=1)
        df_list.append(df)

    data = pd.concat(df_list, ignore_index=True)
    rewards = data["r"].to_numpy()  # episode rewards

    # 3) Compute rolling mean
    rolling_mean = pd.Series(rewards).rolling(window, min_periods=1).mean().to_numpy()
    episodes = np.arange(1, len(rolling_mean) + 1)

    # 4) Plot
    fig, ax = plt.subplots()
    ax.plot(episodes, rolling_mean, label=f"Rolling mean (window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"{case.capitalize()} reward curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
