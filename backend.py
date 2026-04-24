"""
=========================================================================
 backend.py
 FSAE RACK & PINION STATIC STEERING EFFORT PREDICTION TOOL
 
 Handles all physical calculations, data models, file I/O, and console
 reporting. Independent of any GUI or plotting libraries.
=========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np
import pandas as pd

# =========================================================================
# CONSTANTS
# =========================================================================

IN2M = 0.0254          # [in -> m]
LBF2N = 4.44822        # [lbf -> N]
INLB2NM = 0.112985     # [in*lbf -> N*m]
DEG2RAD = np.pi / 180.0

TARGET_MIN_NM = 6.0
TARGET_MAX_NM = 15.0

# =========================================================================
# CALCULATION DATA MODELS
# =========================================================================

@dataclass(frozen=True)
class SteeringConfig:
    """Vehicle/setup inputs for one steering effort calculation."""

    MU: float
    W_CAR_LBF: float
    FRONT_DIST: float
    SCRUB_RADIUS_IN: float
    MECH_TRAIL_IN: float
    PNEUM_TRAIL_IN: float
    CASTER_ANGLE_DEG: float
    KPI_ANGLE_DEG: float
    TOE_LINK_ANGLE_DEG: float
    PINION_RADIUS_IN: float
    SW_RADIUS_IN: float
    TIEROD_MOMENT_ARM_IN: float
    COL_FRICTION_INLB: float
    KP_PATCH_STATIC_IN: float
    KP_PATCH_DYNAMIC_IN: float
    MAX_SW_ANGLE_DEG: float = 180.0
    N_STEPS: int = 1000
    name: str = "Setup"

    @classmethod
    def from_dict(cls, data: dict[str, Any], name: str | None = None) -> "SteeringConfig":
        cleaned = dict(data)
        cfg_name = name or cleaned.pop("_name", cleaned.pop("name", "Setup"))
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        allowed.discard("name")
        values = {key: cleaned[key] for key in allowed if key in cleaned}
        values["name"] = cfg_name

        missing = [key for key in required_config_keys() if key not in values]
        if missing:
            raise KeyError(f"Missing required config value(s): {', '.join(missing)}")
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path, name: str | None = None) -> "SteeringConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, name=name or path.stem)

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["_name"] = data.pop("name")
        return data


def required_config_keys() -> list[str]:
    return [
        "MU", "W_CAR_LBF", "FRONT_DIST", "SCRUB_RADIUS_IN", "MECH_TRAIL_IN",
        "PNEUM_TRAIL_IN", "CASTER_ANGLE_DEG", "KPI_ANGLE_DEG", "TOE_LINK_ANGLE_DEG",
        "PINION_RADIUS_IN", "SW_RADIUS_IN", "TIEROD_MOMENT_ARM_IN", "COL_FRICTION_INLB",
        "KP_PATCH_STATIC_IN", "KP_PATCH_DYNAMIC_IN",
    ]


@dataclass(frozen=True)
class SteeringResults:
    """Calculated result arrays for one steering setup."""
    config: SteeringConfig
    rack_speed_in_per_rev: float
    w_front_lbf: float
    w_per_front_lbf: float
    max_lateral_tire_force_per_corner_lbf: float
    sw_angles_deg: np.ndarray
    wheel_angle_deg: np.ndarray
    rack_disp_in: np.ndarray
    kp_patch_in: np.ndarray
    tierod_arm_eff_in: np.ndarray
    T_trail_Nm: np.ndarray
    T_scrub_Nm: np.ndarray
    T_caster_Nm: np.ndarray
    T_kpi_Nm: np.ndarray
    T_align_Nm: np.ndarray
    T_geom_Nm: np.ndarray
    T_pinion_Nm: np.ndarray
    T_col_Nm: float
    T_total_Nm: np.ndarray
    F_rack_N: np.ndarray

    def as_dataframe(self) -> pd.DataFrame:
        n = len(self.sw_angles_deg)
        return pd.DataFrame({
            "SW_Angle_deg": self.sw_angles_deg,
            "Wheel_Angle_deg": self.wheel_angle_deg,
            "Rack_Disp_in": self.rack_disp_in,
            "KP_Patch_in": self.kp_patch_in,
            "TieRod_Arm_Eff_in": self.tierod_arm_eff_in,
            "T_Trail_Nm": self.T_trail_Nm,
            "T_Scrub_Nm": self.T_scrub_Nm,
            "T_Caster_Nm": self.T_caster_Nm,
            "T_KPI_Nm": self.T_kpi_Nm,
            "T_Align_Nm": self.T_align_Nm,
            "T_Geom_Nm": self.T_geom_Nm,
            "T_Pinion_Nm": self.T_pinion_Nm,
            "T_ColFriction_Nm": np.full(n, self.T_col_Nm),
            "T_Total_Nm": self.T_total_Nm,
            "F_Rack_N": self.F_rack_N,
        })

    def nearest_index_by_sw_angle(self, sw_angle_deg: float) -> int:
        return int(np.argmin(np.abs(self.sw_angles_deg - sw_angle_deg)))

    def nearest_index_by_wheel_angle(self, wheel_angle_deg: float) -> int:
        return int(np.argmin(np.abs(self.wheel_angle_deg - wheel_angle_deg)))


# =========================================================================
# PURE CALCULATION LOGIC
# =========================================================================

def validate_config(cfg: SteeringConfig) -> None:
    if cfg.N_STEPS < 2: raise ValueError("N_STEPS must be at least 2.")
    if cfg.MAX_SW_ANGLE_DEG <= 0: raise ValueError("MAX_SW_ANGLE_DEG must be positive.")
    if cfg.PINION_RADIUS_IN <= 0: raise ValueError("PINION_RADIUS_IN must be positive.")
    if cfg.TIEROD_MOMENT_ARM_IN <= 0: raise ValueError("TIEROD_MOMENT_ARM_IN must be positive.")
    if cfg.SW_RADIUS_IN <= 0: raise ValueError("SW_RADIUS_IN must be positive.")
    if not 0 <= cfg.FRONT_DIST <= 1: raise ValueError("FRONT_DIST should be a fraction between 0 and 1.")
    if cfg.W_CAR_LBF <= 0: raise ValueError("W_CAR_LBF must be positive.")
    if cfg.MU < 0: raise ValueError("MU cannot be negative.")
    if abs(np.cos(cfg.TOE_LINK_ANGLE_DEG * DEG2RAD)) < 1e-6:
        raise ValueError("TOE_LINK_ANGLE_DEG is too close to 90 degrees.")

def calculate_steering_effort(cfg: SteeringConfig) -> SteeringResults:
    validate_config(cfg)
    w_front_lbf = cfg.W_CAR_LBF * cfg.FRONT_DIST
    w_per_front_lbf = w_front_lbf / 2.0
    rack_speed_in_per_rev = 2.0 * np.pi * cfg.PINION_RADIUS_IN
    max_lateral_tire_force_per_corner_lbf = cfg.MU * w_per_front_lbf

    sw_angles_deg = np.linspace(0.0, cfg.MAX_SW_ANGLE_DEG, int(cfg.N_STEPS))
    sw_angles_rad = sw_angles_deg * DEG2RAD
    rack_disp_in = sw_angles_rad * cfg.PINION_RADIUS_IN
    arcsin_arg = np.clip(rack_disp_in / cfg.TIEROD_MOMENT_ARM_IN, -0.9999, 0.9999)
    wheel_angle_rad = np.arcsin(arcsin_arg)
    wheel_angle_deg = wheel_angle_rad / DEG2RAD

    kp_patch_in = cfg.KP_PATCH_STATIC_IN + (
        (cfg.KP_PATCH_DYNAMIC_IN - cfg.KP_PATCH_STATIC_IN) * (wheel_angle_deg / 30.0)
    )
    kp_patch_in = np.maximum(kp_patch_in, cfg.KP_PATCH_STATIC_IN)

    caster_rad = cfg.CASTER_ANGLE_DEG * DEG2RAD
    kpi_rad = cfg.KPI_ANGLE_DEG * DEG2RAD
    toe_link_rad = cfg.TOE_LINK_ANGLE_DEG * DEG2RAD
    total_trail_in = cfg.MECH_TRAIL_IN + cfg.PNEUM_TRAIL_IN

    T_trail_inlb = cfg.MU * w_per_front_lbf * total_trail_in * np.cos(wheel_angle_rad)
    T_scrub_inlb = cfg.MU * w_per_front_lbf * cfg.SCRUB_RADIUS_IN * np.sin(wheel_angle_rad)
    T_align_inlb = T_trail_inlb + T_scrub_inlb

    T_caster_inlb = w_per_front_lbf * cfg.MECH_TRAIL_IN * np.sin(wheel_angle_rad) * np.sin(caster_rad)
    T_kpi_inlb = w_per_front_lbf * cfg.SCRUB_RADIUS_IN * np.sin(wheel_angle_rad) * np.sin(kpi_rad) * np.cos(caster_rad)
    T_geom_inlb = T_caster_inlb + T_kpi_inlb

    toe_link_factor = 1.0 / np.cos(toe_link_rad)
    T_kp_total_inlb = T_align_inlb + T_geom_inlb

    tierod_arm_eff_in = cfg.TIEROD_MOMENT_ARM_IN * np.cos(wheel_angle_rad)
    F_tierod_lbf = T_kp_total_inlb / tierod_arm_eff_in
    F_rack_lbf = F_tierod_lbf * toe_link_factor
    T_pinion_inlb = F_rack_lbf * cfg.PINION_RADIUS_IN

    T_total_inlb = T_pinion_inlb + cfg.COL_FRICTION_INLB

    return SteeringResults(
        config=cfg,
        rack_speed_in_per_rev=rack_speed_in_per_rev,
        w_front_lbf=w_front_lbf,
        w_per_front_lbf=w_per_front_lbf,
        max_lateral_tire_force_per_corner_lbf=max_lateral_tire_force_per_corner_lbf,
        sw_angles_deg=sw_angles_deg,
        wheel_angle_deg=wheel_angle_deg,
        rack_disp_in=rack_disp_in,
        kp_patch_in=kp_patch_in,
        tierod_arm_eff_in=tierod_arm_eff_in,
        T_trail_Nm=T_trail_inlb * INLB2NM,
        T_scrub_Nm=T_scrub_inlb * INLB2NM,
        T_caster_Nm=T_caster_inlb * INLB2NM,
        T_kpi_Nm=T_kpi_inlb * INLB2NM,
        T_align_Nm=T_align_inlb * INLB2NM,
        T_geom_Nm=T_geom_inlb * INLB2NM,
        T_pinion_Nm=T_pinion_inlb * INLB2NM,
        T_col_Nm=cfg.COL_FRICTION_INLB * INLB2NM,
        T_total_Nm=T_total_inlb * INLB2NM,
        F_rack_N=F_rack_lbf * LBF2N,
    )

def calculate_many(configs: Iterable[SteeringConfig]) -> list[SteeringResults]:
    return [calculate_steering_effort(cfg) for cfg in configs]

def run_sensitivity_sweep(cfg: SteeringConfig, ref_sw_angle_deg: float = 105.0, delta_pct: float = 0.25, n_points: int = 20) -> dict[str, pd.DataFrame]:
    validate_config(cfg)
    sweep_specs = {
        "Caster Angle [deg]": "CASTER_ANGLE_DEG",
        "KPI Angle [deg]": "KPI_ANGLE_DEG",
        "Scrub Radius [in]": "SCRUB_RADIUS_IN",
        "Mechanical Trail [in]": "MECH_TRAIL_IN",
        "Tie-rod Moment Arm [in]": "TIEROD_MOMENT_ARM_IN",
        "Coefficient of Friction": "MU",
    }
    output: dict[str, pd.DataFrame] = {}

    for label, attr in sweep_specs.items():
        base_val = float(getattr(cfg, attr))
        sweep_vals = np.linspace(base_val * (1 - delta_pct), base_val * (1 + delta_pct), n_points)
        efforts = []

        for sweep_val in sweep_vals:
            cfg_dict = asdict(cfg)
            cfg_dict[attr] = float(sweep_val)
            swept_cfg = SteeringConfig(**cfg_dict)
            result = calculate_steering_effort(swept_cfg)
            idx = result.nearest_index_by_sw_angle(ref_sw_angle_deg)
            efforts.append(float(result.T_total_Nm[idx]))

        output[label] = pd.DataFrame({
            "parameter_value": sweep_vals,
            "parameter_multiple_of_baseline": sweep_vals / base_val,
            "effort_Nm": efforts,
        })
    return output

# =========================================================================
# FILE I/O HELPERS
# =========================================================================

def load_baseline_config(path: str | Path = "setup_baseline.json") -> SteeringConfig:
    return SteeringConfig.from_json(path, name="Manual Setup Override")

def export_results_csv(result: SteeringResults, output_dir: str | Path = ".") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(result.config.name)
    path = output_dir / f"steering_effort_results_{safe_name}.csv"
    result.as_dataframe().to_csv(path, index=False)
    return path

def sanitize_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace() or ch in (".", "/", "\\"):
            keep.append("_")
    return "".join(keep).strip("_") or "setup"

# =========================================================================
# CONSOLE REPORTING
# =========================================================================

def print_result_summary(result: SteeringResults) -> None:
    cfg = result.config
    print(f"\n{'=' * 50}")
    print(f" PROCESSING: {cfg.name}")
    print(f"{'=' * 50}")

    print("=== FSAE Steering Effort Tool - Derived Parameters ===")
    print(f"  Rack speed                   : {result.rack_speed_in_per_rev:.4f} in/rev")
    print(f"  Front corner load            : {result.w_per_front_lbf:.2f} lbf")
    print(f"  Max lateral tyre force/corner: {result.max_lateral_tire_force_per_corner_lbf:.2f} lbf  (mu x W)")

    print("\n=== Steering Effort Summary at Key Wheel Angles ===")
    header = (
        f"{'SW [deg]':<12} {'Wheel [deg]':<12} {'T_align [Nm]':<14}"
        f"{'T_geom [Nm]':<13} {'T_col [Nm]':<12} {'T_total [Nm]':<13} {'F_rack [N]':<10}"
    )
    print(header)
    print("-" * len(header))

    check_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 150, 180]
    for ca in check_angles:
        idx = result.nearest_index_by_sw_angle(ca)
        print(
            f"{result.sw_angles_deg[idx]:<12.1f} {result.wheel_angle_deg[idx]:<12.2f}"
            f"{result.T_align_Nm[idx]:<14.3f} {result.T_geom_Nm[idx]:<13.3f}"
            f"{result.T_col_Nm:<12.3f} {result.T_total_Nm[idx]:<13.3f} {result.F_rack_N[idx]:<10.2f}"
        )

    idx_30 = result.nearest_index_by_wheel_angle(30.0)
    print("\n=== Component Breakdown at ~30 deg Wheel Angle ===")
    print(f"  SW angle at 30 deg wheel : {result.sw_angles_deg[idx_30]:.1f} deg")
    print(f"  Trail torque             : {result.T_trail_Nm[idx_30]:+.3f} N*m")
    print(f"  Scrub torque             : {result.T_scrub_Nm[idx_30]:+.3f} N*m")
    print(f"  Caster restoration       : {result.T_caster_Nm[idx_30]:+.3f} N*m")
    print(f"  KPI restoration          : {result.T_kpi_Nm[idx_30]:+.3f} N*m")
    print(f"  Sub-total at KP axis     : {(result.T_align_Nm[idx_30] + result.T_geom_Nm[idx_30]):+.3f} N*m")
    print(f"  Referred to pinion       : {result.T_pinion_Nm[idx_30]:+.3f} N*m")
    print(f"  Column friction          : {result.T_col_Nm:+.3f} N*m")
    print(f"  TOTAL steering effort    : {result.T_total_Nm[idx_30]:+.3f} N*m")

    print("\n=== TARGET VALIDATION ===")
    for label, sw_angle in [("0 deg SW", 0.0), ("105 deg SW", 105.0)]:
        idx = result.nearest_index_by_sw_angle(sw_angle)
        effort = result.T_total_Nm[idx]
        status = (
            f"[OK: Within {TARGET_MIN_NM}-{TARGET_MAX_NM} N*m target]"
            if TARGET_MIN_NM <= effort <= TARGET_MAX_NM
            else f"[WARN: Outside {TARGET_MIN_NM}-{TARGET_MAX_NM} N*m target]"
        )
        print(f"  Effort at {label:<10}: {effort:.3f} N*m  {status}")