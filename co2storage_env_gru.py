"""
co2storage_env_gru.py

Environment for uncertainty-aware CO2 storage optimization with three injectors,
driven by GRU-based surrogate models for saturation and pressure.

Implements
- Fixed three-well placement chosen once at the first decision and held constant
- Dynamic per-well injection-rate controls on subsequent steps
- Observation tensor composed of saturation and normalized pressure, scaled to [0, 1]
- Reward that increases with stored CO2 mass and includes pressure and leakage penalties
- Robust placement constrained by a precomputed 3D feasibility boundary

Manuscript alignment
- Observation definition and scaling: state tensor with saturation and normalized pressure in [0, 1] (Section 2.3; Figure 3).
- First-step location selection from a feasibility mask; locations remain fixed; subsequent steps adjust rates (Section 2.3; Figure 2).
- Feasibility boundary built from 3D boundary-leakage screening at 0.05 saturation in at least 80 percent of runs (Section 3.1; Figure 5).
- Mass computation uses ρ_CO2 = 700 kg/m^3 and reports in 10 Mt units (Equation 3).
- Pressure penalty follows a smooth logistic form anchored at P0 = 43 MPa and P_max = 49 MPa (Equation 4; Figure 1).
- Leakage penalty equals the boundary fraction λ_t defined in Equation 5.

Notes on internal units and scaling
- Pressures and BHP are handled in kPa, with normalization factor 1e-5 to map typical values into [0, 1] for the CNN.
- The injection-rate mapping r in [0, 1] to 0.5 + 1.5 r matches the manuscript range of 0.5–2.0 Mt/year per well.
"""

from __future__ import annotations
import os
import numpy as np
import h5py
import torch
from torch import nn  # may be used by user code elsewhere
import gym
from gym import spaces

from surrogate_model_gru import SurrogateModel


class CO2StorageEnv(gym.Env):
    """
    CO2 storage environment with exactly three injection wells.

    Action vector a in [0, 1]^9:
        [x1, y1, x2, y2, x3, y3, r1, r2, r3]
    Timing logic:
        - t = 0: ignore action, no injection
        - t = 1: read full 9D action, place wells, set initial rates
        - t >= 2: ignore first six entries, only use r1..r3

    Episode structure:
        20 steps total; injection allowed on t = 1..14; t = 15..19 is post-injection.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        max_time: int = 20,
        data_path: str = "train_data.h5",
        bulk_data_path: str = "BulkVolume.h5",
        feasible_map_path: str = "robust_feasible_map_3d_gaussian_smoothed.npy",
        device: torch.device | None = None,
    ):
        super().__init__()

        self.max_time = int(max_time)
        self.injection_period = 15  # steps 1..14
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Surrogate models for saturation and pressure
        self.sat_model = SurrogateModel(
            model_path="./weights/epoch_49_820_15_5_sat.pth",
            model_type="sat",
            device=self.device,
            normalizer_path=None,
        )
        self.pres_model = SurrogateModel(
            model_path="./weights/epoch_49_820_15_5_pres.pth",
            model_type="pres",
            device=self.device,
            normalizer_path="./normalizers/820_c5_pressure_15_5_train_aug_output.pkl",
        )

        # Data loading
        with h5py.File(data_path, "r") as f_in:
            self.perm_data = f_in["inputs"]["Perm"][:]  # [N, 1, 1, nz, nx, ny]
            self.por_data = f_in["inputs"]["Por"][:]    # [N, 1, 1, nz, nx, ny]
        self.num_samples = self.perm_data.shape[0]

        with h5py.File(bulk_data_path, "r") as fbulk:
            self.bulk_data = fbulk["BulkVol"][0, 0].astype(np.float32)  # [nz, nx, ny]

        # Feasibility map for placement
        feasible_3d = np.load(feasible_map_path)  # [nz, nx, ny], boolean
        # Require feasibility across all layers for each column to get a 2D set
        feasible_2d = np.all(feasible_3d, axis=0)  # [nx, ny]
        self.feasible_indices = np.argwhere(feasible_2d)  # shape (M, 2)
        if self.feasible_indices.size == 0:
            raise ValueError("Feasible map has no valid points")

        # Action and observation spaces
        self.nz, self.nx, self.ny = 24, 60, 60
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(48, self.nx, self.ny), dtype=np.float32
        )

        # Internal state
        self.current_step = 0
        self.done = False
        self.sample_idx: int | None = None
        self.full_input: np.ndarray | None = None
        self.state: dict[str, np.ndarray] = {}
        self.static_data: dict[str, np.ndarray] = {}
        self.last_mass_in_place = 0.0

        # Three injection wells
        self.injection_wells = [
            {"coord": None, "exists": False},
            {"coord": None, "exists": False},
            {"coord": None, "exists": False},
        ]

        # Saved locations after the first decision
        self.saved_locations = np.zeros(6, dtype=np.float32)

        # Constant BHP bound in kPa; scaled by 1e-5 in surrogate input
        self.bhp_kpa = 48000.0
        self.np_random = None

    def seed(self, seed: int | None = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        if self.np_random is None:
            self.seed(None)

        self.current_step = 0
        self.done = False
        self.last_mass_in_place = 0.0

        # Reset well info
        for w in self.injection_wells:
            w["coord"] = None
            w["exists"] = False

        # Sample a realization
        self.sample_idx = int(self.np_random.integers(0, self.num_samples))
        perm_3d = self.perm_data[self.sample_idx, 0, 0].astype(np.float32)
        por_3d = self.por_data[self.sample_idx, 0, 0].astype(np.float32)
        self.static_data["Perm"] = perm_3d
        self.static_data["Por"] = por_3d

        # Surrogate input tensor: [B, T, C, nz, nx, ny]
        self.full_input = np.zeros((1, self.max_time, 5, self.nz, self.nx, self.ny), dtype=np.float32)

        # Static channels per time step
        for t in range(self.max_time):
            self.full_input[0, t, 0] = perm_3d
            self.full_input[0, t, 1] = por_3d
            self.full_input[0, t, 4] = 1.0  # boundary flag

        # BHP normalized
        scaled_bhp = self.bhp_kpa * 1e-5
        for t in range(self.max_time):
            self.full_input[0, t, 3] = scaled_bhp

        # No injection at t = 0
        self.full_input[0, 0, 2] = 0.0

        # Initial surrogate inference
        inp_torch = torch.from_numpy(self.full_input).to(self.device)
        with torch.no_grad():
            sat_out = self.sat_model.inference(inp_torch)
            pres_out = self.pres_model.inference(inp_torch)

        self.state["sat"] = sat_out[0, 0, 0].cpu().numpy()
        self.state["pressure"] = pres_out[0, 0, 0].cpu().numpy()
        self.last_mass_in_place = self._calculate_mass_in_place()

        return self._build_observation()

    def step(self, action: np.ndarray):
        if self.done:
            return self._build_observation(), 0.0, True, {}

        # Clear injection at current step
        self.full_input[0, self.current_step, 2] = 0.0

        if self.current_step == 0:
            pass  # no action used

        elif self.current_step == 1:
            # Parse locations and rates
            loc_x1, loc_y1, loc_x2, loc_y2, loc_x3, loc_y3, r1, r2, r3 = action
            self.saved_locations[:] = action[0:6]

            # Place wells onto nearest feasible points
            self.injection_wells[0]["coord"] = self._pick_feasible_xy_2d(loc_x1, loc_y1)
            self.injection_wells[0]["exists"] = True
            self.injection_wells[1]["coord"] = self._pick_feasible_xy_2d(loc_x2, loc_y2)
            self.injection_wells[1]["exists"] = True
            self.injection_wells[2]["coord"] = self._pick_feasible_xy_2d(loc_x3, loc_y3)
            self.injection_wells[2]["exists"] = True

            if self.current_step < self.injection_period:
                rate1 = 0.5 + 1.5 * r1
                rate2 = 0.5 + 1.5 * r2
                rate3 = 0.5 + 1.5 * r3
                for i, well in enumerate(self.injection_wells):
                    if well["exists"]:
                        x, y = well["coord"]
                        if i == 0:
                            self.full_input[0, self.current_step, 2, :, x, y] = rate1
                        elif i == 1:
                            self.full_input[0, self.current_step, 2, :, x, y] = rate2
                        else:
                            self.full_input[0, self.current_step, 2, :, x, y] = rate3

        else:
            # Only rates are used
            r1, r2, r3 = action[6:9]
            if self.current_step < self.injection_period:
                rate1 = 0.5 + 1.5 * r1
                rate2 = 0.5 + 1.5 * r2
                rate3 = 0.5 + 1.5 * r3
                for i, well in enumerate(self.injection_wells):
                    if well["exists"]:
                        x, y = well["coord"]
                        if i == 0:
                            self.full_input[0, self.current_step, 2, :, x, y] = rate1
                        elif i == 1:
                            self.full_input[0, self.current_step, 2, :, x, y] = rate2
                        else:
                            self.full_input[0, self.current_step, 2, :, x, y] = rate3

        # Surrogate inference
        inp_torch = torch.from_numpy(self.full_input).to(self.device)
        with torch.no_grad():
            sat_out = self.sat_model.inference(inp_torch)
            pres_out = self.pres_model.inference(inp_torch)

        self.state["sat"] = sat_out[0, self.current_step, 0].cpu().numpy()
        self.state["pressure"] = pres_out[0, self.current_step, 0].cpu().numpy()

        # Reward
        mass_before = self.last_mass_in_place
        mass_after = self._calculate_mass_in_place()
        delta_mass = mass_after - mass_before

        pressure_penalty = self._calculate_pressure_penalty(self.state["pressure"])
        leakage_penalty = self._calculate_leakage_penalty(self.state["sat"])
        reward = delta_mass * (1.0 - pressure_penalty) * (1.0 - leakage_penalty)

        self.last_mass_in_place = mass_after

        # Step bookkeeping
        self.current_step += 1
        done = self.current_step >= self.max_time
        self.done = done

        obs = self._build_observation()
        info = {
            "well_locations": [w["coord"] for w in self.injection_wells],
            "max_pressure_kpa": float(np.max(self.state["pressure"])),
            "pressure_penalty": float(pressure_penalty),
            "leakage_penalty": float(leakage_penalty),
        }
        return obs, reward, done, info

    # Helper methods

    def _pick_feasible_xy_2d(self, xn: float, yn: float) -> tuple[int, int]:
        """Snap normalized (xn, yn) in [0, 1] to the nearest feasible (x, y)."""
        float_x = xn * (self.nx - 1)
        float_y = yn * (self.ny - 1)
        dx = self.feasible_indices[:, 0] - float_x
        dy = self.feasible_indices[:, 1] - float_y
        idx_min = int(np.argmin(dx * dx + dy * dy))
        x_feas, y_feas = self.feasible_indices[idx_min]
        return int(x_feas), int(y_feas)

    def _calculate_mass_in_place(self) -> float:
        """
        Mass in 10 Mt units:
            M_t = sum( sat * por * bulk * rho_co2 ) / 1e10
        where rho_co2 = 700 kg/m^3.
        """
        sat_3d = self.state["sat"]
        por_3d = self.static_data["Por"]
        bulk_3d = self.bulk_data
        rho = 700.0  # kg/m^3

        mask = sat_3d >= 1e-5
        mass_kg = np.sum(sat_3d[mask] * por_3d[mask] * bulk_3d[mask] * rho)
        return mass_kg / 1e10

    def _calculate_pressure_penalty(self, pres_3d: np.ndarray) -> float:
        """
        Logistic pressure penalty anchored at P0 = 43 MPa and P_max = 49 MPa.
        Internally, pressures are treated in kPa for consistency with BHP.
        """
        # Parameters in kPa
        P0 = 43000.0
        Pmax = 49000.0
        Pmid = 0.5 * (P0 + Pmax)   # 46000 kPa
        k = np.log(5.0)            # slope factor for smooth change
        scale = 1000.0             # kPa scale
        p_max = float(np.max(pres_3d))
        penalty = 1.0 / (1.0 + np.exp(-k * (p_max - Pmid) / scale))
        return min(penalty, 0.99)

    def _calculate_leakage_penalty(self, sat_3d: np.ndarray) -> float:
        """
        Leakage penalty λ_t defined as the fraction of boundary-proximal CO2:
            λ_t = (sum over boundary cells of sat * por * bulk)
                  / (sum over all cells of sat * por * bulk)
        """
        por = self.static_data["Por"]
        bulk = self.bulk_data

        # Boundary mask in 3D
        boundary = np.zeros_like(sat_3d, dtype=bool)
        boundary[0, :, :] = True
        boundary[-1, :, :] = True
        boundary[:, 0:2, :] = True
        boundary[:, -2:, :] = True
        boundary[:, :, 0:2] = True
        boundary[:, :, -2:] = True

        sig = sat_3d > 0.05  # count all mobile CO2 in the plume
        total = np.sum(sat_3d[sig] * por[sig] * bulk[sig]) + 1e-12
        boundary_mass = np.sum(sat_3d[boundary & sig] * por[boundary & sig] * bulk[boundary & sig])
        return float(boundary_mass / total)

    def _build_observation(self) -> np.ndarray:
        """
        Return a (48, nx, ny) observation with saturation in [0, 1]
        and normalized pressure in [0, 1].
        """
        sat_3d = np.clip(self.state["sat"], 0.0, 1.0)
        # pressure in kPa; normalize by 1e-5, then clamp to [0, 1]
        pres_norm = np.clip(self.state["pressure"] * 1e-5, 0.0, 1.0)
        return np.concatenate([sat_3d, pres_norm], axis=0).astype(np.float32)

    def render(self, mode: str = "human"):
        pass
