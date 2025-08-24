"""
CMA-ES baseline for GCS — manuscript-accurate implementation.

This script reproduces the manuscript's CMA-ES baseline exactly:
  • Decision dimension d = 48 (6 well-location coords + 14×3 rate controls)
  • Population size = 96 (2d heuristic)
  • Generations per realization = 100
  • Validation realizations = 20 (indices 0..19)
  • 8 CPU cores dedicated to parallel rollout evaluation (per generation)
  • Sample-wise optimization (a separate CMA-ES run per realization)

Outputs:
  • cma_live_log.csv  with rows: [Sample, Reward, Genome48]
"""

import os
import time
import random
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import cma
import torch

# Import your environment class (adjust module path if needed)
from CO2envCS import CO2StorageEnv

# ----------------------- Reproducibility & CPU settings -----------------------
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

# Manuscript: 8 CPU cores dedicated to parallel environment rollouts
N_WORKERS = 8
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

DEVICE = torch.device("cpu")  # manuscript baseline uses CPU workers

# ----------------------------- File sanity checks -----------------------------
REQUIRED = [
    "valid_data.h5",
    "BulkVolume.h5",
    "robust_feasible_map_3d_gaussian_smoothed.npy",
    "./weights/epoch_49_820_15_5_sat.pth",
    "./weights/epoch_49_820_15_5_pres.pth",
]
for f in REQUIRED:
    if not Path(f).is_file():
        raise FileNotFoundError(f"Missing file: {f}")

# ------------------------------ Environment args ------------------------------
ENV_KWARGS = dict(
    max_time=20,
    data_path="valid_data.h5",
    bulk_data_path="BulkVolume.h5",
    feasible_map_path="robust_feasible_map_3d_gaussian_smoothed.npy",
    device=DEVICE,
)

# ---------------------- Genome (48d) → 20-step action schedule ----------------
def vector_to_action_schedule(vec48: np.ndarray):
    """
    48-d vector in [0,1] → 20-step 9-d actions:
      t=0:  no injection
      t=1:  6D locations + first 3D rate triple
      t=2..14: only 3D rates (locations fixed)
      t=15..19: post-injection (zeros)
    """
    v = np.clip(np.asarray(vec48, dtype=np.float32), 0.0, 1.0)
    loc_part, rate_parts = v[:6], v[6:].reshape(14, 3)
    sched = []
    for t in range(20):
        if t == 0:
            sched.append(np.zeros(9, np.float32))
        elif t == 1:
            sched.append(np.concatenate([loc_part, rate_parts[0]], dtype=np.float32))
        elif 2 <= t <= 14:
            sched.append(np.concatenate([np.zeros(6, np.float32), rate_parts[t-1]], dtype=np.float32))
        else:
            sched.append(np.zeros(9, np.float32))
    return sched

def run_episode(env: CO2StorageEnv, sample_idx: int, vec48: np.ndarray) -> float:
    obs = env.reset(sample_idx=sample_idx)
    total = 0.0
    for act in vector_to_action_schedule(vec48):
        obs, r, done, _ = env.step(act)
        total += float(r)
        if done:
            break
    return total

# -------------------------- Multiprocessing evaluation ------------------------
# Each worker holds its own env instance to avoid cross-talk.
_ENV = None

def _worker_init(env_kwargs):
    global _ENV
    torch.set_num_threads(1)
    _ENV = CO2StorageEnv(**env_kwargs)

def _objective_for_tuple(args):
    """
    args: (sample_idx, vector48) → returns negative reward (CMA minimizes)
    """
    sample_idx, vec = args
    return -run_episode(_ENV, sample_idx, vec)

# --------------------------- CMA-ES per-realization run -----------------------
def optimise_cma_single_sample(sample_idx: int,
                               env_kwargs: dict,
                               popsize=96,
                               max_iter=100,
                               sigma0=0.25,
                               seed=GLOBAL_SEED):
    """
    CMA-ES on one fixed realization (sample_idx).
    Returns (best_vector, best_reward).
    """
    x0 = np.full(48, 0.1, np.float32)
    es = cma.CMAEvolutionStrategy(
        x0.tolist(), sigma0,
        dict(popsize=popsize, bounds=[0, 1], seed=seed)
    )

    best_r, best_x = -np.inf, None

    # Pool per sample—parallel fitness on 8 CPU workers
    with mp.Pool(processes=N_WORKERS, initializer=_worker_init, initargs=(env_kwargs,)) as pool:
        for _ in range(max_iter):
            sols = es.ask()
            args = [(sample_idx, np.asarray(s, np.float32)) for s in sols]
            fits = pool.map(_objective_for_tuple, args)  # parallel evaluation
            es.tell(sols, fits)

            i_best = int(np.argmin(fits))
            cand_r = -float(fits[i_best])
            if cand_r > best_r:
                best_r = cand_r
                best_x = np.asarray(sols[i_best], np.float32)

            if es.stop():
                break

    return best_x, float(best_r)

# ----------------------------------- Main -------------------------------------
def main():
    mp.set_start_method("spawn", force=True)

    # Manuscript: 20 validation realizations, 96 pop, 100 generations
    VALID_SAMPLES = list(range(20))
    POP_SIZE = 96
    MAX_ITER = 100
    RESULT_CSV = "cma_live_log.csv"

    # Prepare results file
    if not Path(RESULT_CSV).is_file():
        pd.DataFrame(columns=["Sample", "Reward", "Genome48"]).to_csv(RESULT_CSV, index=False)

    def already_done(sample):
        try:
            df = pd.read_csv(RESULT_CSV)
            return (df["Sample"] == sample).any()
        except Exception:
            return False

    print("CMA-ES baseline (GCS) — 8 CPU workers | pop=96 | gens=100 | 20 validations")
    for s in tqdm(VALID_SAMPLES, desc="CMA-ES validations"):
        if already_done(s):
            print(f"[Sample {s}] already logged — skip")
            continue

        t0 = time.time()
        x_best, r_best = optimise_cma_single_sample(
            sample_idx=s,
            env_kwargs=ENV_KWARGS,
            popsize=POP_SIZE,
            max_iter=MAX_ITER,
            sigma0=0.25,
            seed=GLOBAL_SEED
        )
        mins = (time.time() - t0) / 60.0
        print(f"[Sample {s}] best reward = {r_best:.4f} | {mins:.1f} min")

        pd.DataFrame([[s, r_best, x_best.tolist()]],
                     columns=["Sample", "Reward", "Genome48"]).to_csv(
            RESULT_CSV, mode="a", header=False, index=False
        )

    print("CMA-ES sweep complete →", RESULT_CSV)

if __name__ == "__main__":
    main()
