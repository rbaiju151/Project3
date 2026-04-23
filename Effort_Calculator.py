"""
=========================================================================
 FSAE RACK & PINION STATIC STEERING EFFORT PREDICTION TOOL
=========================================================================
 Refactored version: calculation logic is independent from GUI, plotting,
 console printing, and file export.

 Units:
 - Config inputs are US customary unless noted by key name.
 - Calculations are mostly performed in in, lbf, in*lbf.
 - Public result arrays are exported in SI where practical.
=========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
import glob
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox


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

    # Tire / road
    MU: float

    # Vehicle mass and distribution
    W_CAR_LBF: float
    FRONT_DIST: float

    # Steering geometry
    SCRUB_RADIUS_IN: float
    MECH_TRAIL_IN: float
    PNEUM_TRAIL_IN: float
    CASTER_ANGLE_DEG: float
    KPI_ANGLE_DEG: float
    TOE_LINK_ANGLE_DEG: float

    # Steering rack / column
    PINION_RADIUS_IN: float
    SW_RADIUS_IN: float
    TIEROD_MOMENT_ARM_IN: float
    COL_FRICTION_INLB: float

    # Kingpin axis to contact-patch distances
    KP_PATCH_STATIC_IN: float
    KP_PATCH_DYNAMIC_IN: float

    # Steering travel
    MAX_SW_ANGLE_DEG: float = 180.0
    N_STEPS: int = 1000

    # Metadata
    name: str = "Setup"

    @classmethod
    def from_dict(cls, data: dict[str, Any], name: str | None = None) -> "SteeringConfig":
        """Build a strongly typed config from a JSON-style dict."""
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
        """Load a config from a JSON file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, name=name or path.stem)

    def to_json_dict(self) -> dict[str, Any]:
        """Return config as a JSON-serializable dictionary."""
        data = asdict(self)
        data["_name"] = data.pop("name")
        return data


def required_config_keys() -> list[str]:
    """Required JSON keys, excluding optional travel parameters and metadata."""
    return [
        "MU",
        "W_CAR_LBF",
        "FRONT_DIST",
        "SCRUB_RADIUS_IN",
        "MECH_TRAIL_IN",
        "PNEUM_TRAIL_IN",
        "CASTER_ANGLE_DEG",
        "KPI_ANGLE_DEG",
        "TOE_LINK_ANGLE_DEG",
        "PINION_RADIUS_IN",
        "SW_RADIUS_IN",
        "TIEROD_MOMENT_ARM_IN",
        "COL_FRICTION_INLB",
        "KP_PATCH_STATIC_IN",
        "KP_PATCH_DYNAMIC_IN",
    ]


@dataclass(frozen=True)
class SteeringResults:
    """Calculated result arrays for one steering setup."""

    config: SteeringConfig

    # Derived scalar parameters
    rack_speed_in_per_rev: float
    w_front_lbf: float
    w_per_front_lbf: float
    max_lateral_tire_force_per_corner_lbf: float

    # Kinematics
    sw_angles_deg: np.ndarray
    wheel_angle_deg: np.ndarray
    rack_disp_in: np.ndarray
    kp_patch_in: np.ndarray
    tierod_arm_eff_in: np.ndarray

    # Torque components, SI
    T_trail_Nm: np.ndarray
    T_scrub_Nm: np.ndarray
    T_caster_Nm: np.ndarray
    T_kpi_Nm: np.ndarray
    T_align_Nm: np.ndarray
    T_geom_Nm: np.ndarray
    T_pinion_Nm: np.ndarray
    T_col_Nm: float
    T_total_Nm: np.ndarray

    # Forces, SI
    F_rack_N: np.ndarray

    def as_dataframe(self) -> pd.DataFrame:
        """Return the main calculation sweep as a pandas DataFrame."""
        n = len(self.sw_angles_deg)
        return pd.DataFrame(
            {
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
            }
        )

    def nearest_index_by_sw_angle(self, sw_angle_deg: float) -> int:
        return int(np.argmin(np.abs(self.sw_angles_deg - sw_angle_deg)))

    def nearest_index_by_wheel_angle(self, wheel_angle_deg: float) -> int:
        return int(np.argmin(np.abs(self.wheel_angle_deg - wheel_angle_deg)))


# =========================================================================
# PURE CALCULATION LOGIC
# =========================================================================


def validate_config(cfg: SteeringConfig) -> None:
    """Validate basic physical and numerical assumptions before calculation."""
    if cfg.N_STEPS < 2:
        raise ValueError("N_STEPS must be at least 2.")
    if cfg.MAX_SW_ANGLE_DEG <= 0:
        raise ValueError("MAX_SW_ANGLE_DEG must be positive.")
    if cfg.PINION_RADIUS_IN <= 0:
        raise ValueError("PINION_RADIUS_IN must be positive.")
    if cfg.TIEROD_MOMENT_ARM_IN <= 0:
        raise ValueError("TIEROD_MOMENT_ARM_IN must be positive.")
    if cfg.SW_RADIUS_IN <= 0:
        raise ValueError("SW_RADIUS_IN must be positive.")
    if not 0 <= cfg.FRONT_DIST <= 1:
        raise ValueError("FRONT_DIST should be a fraction between 0 and 1.")
    if cfg.W_CAR_LBF <= 0:
        raise ValueError("W_CAR_LBF must be positive.")
    if cfg.MU < 0:
        raise ValueError("MU cannot be negative.")
    if abs(np.cos(cfg.TOE_LINK_ANGLE_DEG * DEG2RAD)) < 1e-6:
        raise ValueError("TOE_LINK_ANGLE_DEG is too close to 90 degrees.")


def calculate_steering_effort(cfg: SteeringConfig) -> SteeringResults:
    """
    Calculate steering effort for one setup.

    This function is intentionally independent from Tkinter, Matplotlib,
    console printing, and CSV export. It can be unit-tested directly.
    """
    validate_config(cfg)

    w_front_lbf = cfg.W_CAR_LBF * cfg.FRONT_DIST
    w_per_front_lbf = w_front_lbf / 2.0
    rack_speed_in_per_rev = 2.0 * np.pi * cfg.PINION_RADIUS_IN
    max_lateral_tire_force_per_corner_lbf = cfg.MU * w_per_front_lbf

    sw_angles_deg = np.linspace(0.0, cfg.MAX_SW_ANGLE_DEG, int(cfg.N_STEPS))
    sw_angles_rad = sw_angles_deg * DEG2RAD

    rack_disp_in = sw_angles_rad * cfg.PINION_RADIUS_IN

    # Clip to avoid arcsin domain errors when rack travel exceeds the simple
    # tie-rod arm model's mathematical range.
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

    # Contact patch aligning moment.
    T_trail_inlb = cfg.MU * w_per_front_lbf * total_trail_in * np.cos(wheel_angle_rad)
    T_scrub_inlb = cfg.MU * w_per_front_lbf * cfg.SCRUB_RADIUS_IN * np.sin(wheel_angle_rad)
    T_align_inlb = T_trail_inlb + T_scrub_inlb

    # Kingpin geometry restoration torque.
    T_caster_inlb = (
        w_per_front_lbf
        * cfg.MECH_TRAIL_IN
        * np.sin(wheel_angle_rad)
        * np.sin(caster_rad)
    )
    T_kpi_inlb = (
        w_per_front_lbf
        * cfg.SCRUB_RADIUS_IN
        * np.sin(wheel_angle_rad)
        * np.sin(kpi_rad)
        * np.cos(caster_rad)
    )
    T_geom_inlb = T_caster_inlb + T_kpi_inlb

    # Refer kingpin torque through tie-rod and rack to the pinion.
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
    """Calculate steering effort for multiple configs."""
    return [calculate_steering_effort(cfg) for cfg in configs]


def run_sensitivity_sweep(
    cfg: SteeringConfig,
    ref_sw_angle_deg: float = 105.0,
    delta_pct: float = 0.25,
    n_points: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Sweep selected parameters and calculate effort at a reference SW angle.

    This is also calculation-only. Plotting is handled elsewhere.
    """
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

        output[label] = pd.DataFrame(
            {
                "parameter_value": sweep_vals,
                "parameter_multiple_of_baseline": sweep_vals / base_val,
                "effort_Nm": efforts,
            }
        )

    return output


# =========================================================================
# FILE I/O HELPERS
# =========================================================================


def load_baseline_config(path: str | Path = "setup_baseline.json") -> SteeringConfig:
    """Load the baseline setup used to prefill the GUI."""
    return SteeringConfig.from_json(path, name="Manual Setup Override")


def export_results_csv(result: SteeringResults, output_dir: str | Path = ".") -> Path:
    """Export one result set without overwriting other configs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(result.config.name)
    path = output_dir / f"steering_effort_results_{safe_name}.csv"
    result.as_dataframe().to_csv(path, index=False)
    return path


def sanitize_filename(name: str) -> str:
    """Make a simple filesystem-safe filename stem."""
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
    """Print a human-readable summary for one result set."""
    cfg = result.config

    print(f"\n{'=' * 50}")
    print(f" PROCESSING: {cfg.name}")
    print(f"{'=' * 50}")

    print("=== FSAE Steering Effort Tool - Derived Parameters ===")
    print(f"  Rack speed                   : {result.rack_speed_in_per_rev:.4f} in/rev")
    print(f"  Front corner load            : {result.w_per_front_lbf:.2f} lbf")
    print(
        "  Max lateral tyre force/corner: "
        f"{result.max_lateral_tire_force_per_corner_lbf:.2f} lbf  (mu x W)"
    )

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


# =========================================================================
# PLOTTING
# =========================================================================

# Dark theme palette
CLR_BG = (0.08, 0.08, 0.12)
CLR_PANEL = (0.12, 0.12, 0.18)
CLR_TOTAL = (0.20, 0.85, 0.50)
CLR_ALIGN = (0.95, 0.60, 0.10)
CLR_GEOM = (0.30, 0.65, 1.00)
CLR_COL = (0.85, 0.30, 0.30)
CLR_GRID = (0.25, 0.25, 0.30)
CLR_TEXT = (0.92, 0.92, 0.95)


def apply_plot_theme() -> None:
    """Apply global Matplotlib text/tick colors."""
    plt.rcParams.update(
        {
            "text.color": CLR_TEXT,
            "axes.labelcolor": CLR_TEXT,
            "xtick.color": CLR_TEXT,
            "ytick.color": CLR_TEXT,
        }
    )


def style_ax(ax: plt.Axes) -> None:
    """Apply the dark-theme style to one axis."""
    ax.set_facecolor(CLR_PANEL)
    ax.grid(True, color=CLR_GRID, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_color(CLR_TEXT)


def plot_results(result: SteeringResults) -> plt.Figure:
    """Create the main results figure for one result set."""
    apply_plot_theme()
    cfg = result.config

    fig = plt.figure(figsize=(14, 9), facecolor=CLR_BG)
    fig.canvas.manager.set_window_title(f"FSAE Steering Effort Prediction - {cfg.name}")

    sw = result.sw_angles_deg

    # Plot 1: total effort vs steering wheel angle
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    style_ax(ax1)
    ax1.fill_between([0, cfg.MAX_SW_ANGLE_DEG], TARGET_MIN_NM, TARGET_MAX_NM, color=CLR_TOTAL, alpha=0.08, edgecolor="none")
    ax1.axhline(TARGET_MIN_NM, color=(0.50, 0.95, 0.65), linestyle="--", linewidth=1, alpha=0.6, label="6 N*m target min")
    ax1.axhline(TARGET_MAX_NM, color=(0.50, 0.95, 0.65), linestyle="--", linewidth=1, alpha=0.6, label="15 N*m target max")
    ax1.plot(sw, result.T_col_Nm * np.ones_like(sw), "-", color=CLR_COL, linewidth=1.2, label="Column Friction")
    ax1.plot(sw, result.T_align_Nm + result.T_col_Nm, "-", color=CLR_ALIGN, linewidth=1.5, label="+ Aligning Moment")
    ax1.plot(sw, result.T_total_Nm, "-", color=CLR_TOTAL, linewidth=2.5, label="Total Driver Effort")
    ax1.axvline(105, color=CLR_TEXT, linestyle="--", alpha=0.5, linewidth=1, label="105 deg SW limit")
    ax1.set_xlabel("Steering Wheel Angle [deg]", fontsize=11)
    ax1.set_ylabel("Torque [N*m]", fontsize=11)
    ax1.set_title("Total Steering Effort vs. Steering Wheel Angle", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", facecolor=CLR_PANEL, edgecolor=CLR_GRID, fontsize=9, labelcolor=CLR_TEXT)
    ax1.set_xlim([0, cfg.MAX_SW_ANGLE_DEG])
    ax1.set_ylim([0, np.max(result.T_total_Nm) * 1.1])

    # Plot 2: stacked component breakdown
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    style_ax(ax2)
    y1 = result.T_col_Nm * np.ones_like(sw)
    stack_values = [y1, result.T_trail_Nm, result.T_scrub_Nm, result.T_caster_Nm, result.T_kpi_Nm]
    stack_colors = [CLR_COL, (0.95, 0.80, 0.20), (0.85, 0.45, 0.10), (0.30, 0.55, 1.00), (0.60, 0.30, 1.00)]
    stack_labels = ["Column Friction", "Trail Aligning", "Scrub Aligning", "Caster Restore", "KPI Restore"]
    ax2.stackplot(sw, *stack_values, labels=stack_labels, colors=stack_colors, alpha=0.75, edgecolor="none")
    ax2.plot(sw, result.T_total_Nm, "-", color=CLR_TOTAL, linewidth=2, label="Total")
    ax2.set_xlabel("Steering Wheel Angle [deg]", fontsize=10)
    ax2.set_ylabel("Torque [N*m]", fontsize=10)
    ax2.set_title("Stacked Component Breakdown", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", facecolor=CLR_PANEL, edgecolor=CLR_GRID, fontsize=7, labelcolor=CLR_TEXT)
    ax2.set_xlim([0, cfg.MAX_SW_ANGLE_DEG])
    ax2.set_ylim([0, np.max(result.T_total_Nm) * 1.15])

    # Plot 3: wheel angle vs SW angle
    ax3 = plt.subplot2grid((2, 3), (1, 0))
    style_ax(ax3)
    ax3.plot(sw, result.wheel_angle_deg, "-", color=(0.80, 0.70, 1.00), linewidth=2)
    ax3.set_xlabel("Steering Wheel Angle [deg]", fontsize=10)
    ax3.set_ylabel("Wheel Steer Angle [deg]", fontsize=10)
    ax3.set_title("Wheel Angle vs. SW Angle", fontsize=11, fontweight="bold")
    ax3.set_xlim([0, cfg.MAX_SW_ANGLE_DEG])

    # Plot 4: rack force vs SW angle
    ax4 = plt.subplot2grid((2, 3), (1, 1))
    style_ax(ax4)
    ax4.plot(sw, result.F_rack_N, "-", color=(1.0, 0.60, 0.20), linewidth=2)
    ax4.set_xlabel("Steering Wheel Angle [deg]", fontsize=10)
    ax4.set_ylabel("Rack Force [N]", fontsize=10)
    ax4.set_title("Required Rack Force", fontsize=11, fontweight="bold")
    ax4.set_xlim([0, cfg.MAX_SW_ANGLE_DEG])

    # Plot 5: KP-to-patch distance and effective tie-rod arm
    ax5 = plt.subplot2grid((2, 3), (1, 2))
    style_ax(ax5)
    l1 = ax5.plot(result.wheel_angle_deg, result.kp_patch_in * IN2M * 1000, "-", color=(0.40, 0.90, 0.90), linewidth=2, label="KP-Patch Distance")
    ax5.set_ylabel("KP-Patch Dist [mm]", color=(0.40, 0.90, 0.90), fontsize=10)
    ax5.set_xlabel("Wheel Steer Angle [deg]", fontsize=10)
    ax5.tick_params(axis="y", colors=(0.40, 0.90, 0.90))

    ax5_rt = ax5.twinx()
    l2 = ax5_rt.plot(result.wheel_angle_deg, result.tierod_arm_eff_in * IN2M * 1000, "--", color=(1.0, 0.70, 0.40), linewidth=2, label="Effective Tie-rod Arm")
    ax5_rt.set_ylabel("Effective Tie-rod Arm [mm]", color=(1.0, 0.70, 0.40), fontsize=10)
    ax5_rt.tick_params(axis="y", colors=(1.0, 0.70, 0.40))
    for spine in ax5_rt.spines.values():
        spine.set_visible(False)

    ax5.set_title("Geometry Arms vs. Wheel Angle", fontsize=11, fontweight="bold")
    combined_lines = l1 + l2
    combined_labels = [line.get_label() for line in combined_lines]
    ax5.legend(combined_lines, combined_labels, loc="lower right", facecolor=CLR_PANEL, edgecolor=CLR_GRID, fontsize=8, labelcolor=CLR_TEXT)

    fig.suptitle(
        f"[{cfg.name.upper()}] FSAE Steering Effort Prediction\n"
        f"mu={cfg.MU} | W_front={result.w_per_front_lbf:.0f} lbf/corner"
        f" | Caster={cfg.CASTER_ANGLE_DEG} deg | KPI={cfg.KPI_ANGLE_DEG} deg",
        fontsize=14,
        fontweight="bold",
        color=CLR_TEXT,
    )
    plt.tight_layout()
    return fig


def plot_sensitivity(sensitivity: dict[str, pd.DataFrame]) -> plt.Figure:
    """Create sensitivity-analysis figure from calculation-only sweep data."""
    apply_plot_theme()
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), facecolor=CLR_BG)
    fig.canvas.manager.set_window_title("Sensitivity Analysis")
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for sp, (label, df) in enumerate(sensitivity.items()):
        ax = axs.flatten()[sp]
        style_ax(ax)
        ax.plot(
            df["parameter_multiple_of_baseline"],
            df["effort_Nm"],
            "-o",
            color=colors[sp],
            linewidth=2,
            markersize=4,
            markerfacecolor=colors[sp],
        )
        ax.axvline(1.0, color=CLR_TEXT, linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(TARGET_MIN_NM, color=(0.50, 0.95, 0.65), linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(TARGET_MAX_NM, color=(0.50, 0.95, 0.65), linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Parameter Multiple of Baseline", fontsize=9)
        ax.set_ylabel("Effort @ 105 deg SW [N*m]", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")

    fig.suptitle("Sensitivity Analysis - Effort at 105 deg Steering Wheel Angle", fontsize=13, fontweight="bold", color=CLR_TEXT)
    plt.tight_layout()
    return fig


class LineSnapCursor:
    """Native Matplotlib x-axis snapping crosshair and tooltip."""

    def __init__(self, fig: plt.Figure, clr_panel=CLR_PANEL, clr_text=CLR_TEXT, clr_grid=CLR_GRID):
        self.fig = fig
        self.clr_panel = clr_panel
        self.clr_text = clr_text
        self.clr_grid = clr_grid
        self.current_ax = None
        self.v_line = None
        self.h_line = None
        self.annot = None
        # Freeze all axes limits so cursor/annotation artists cannot trigger rescaling.
        for ax in self.fig.axes:
            ax.set_autoscale_on(False)
            ax.set_xlim(ax.get_xlim())
            ax.set_ylim(ax.get_ylim())
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        

    def _setup_cursor_elements(self, ax):
        if self.current_ax == ax:
            return

        self._clear_cursor()
        self.current_ax = ax
        self.v_line = ax.axvline(color=self.clr_text, linestyle=":", alpha=0.6, zorder=100)
        self.h_line = ax.axhline(color=self.clr_text, linestyle=":", alpha=0.6, zorder=100)
        self.annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            annotation_clip=True,
            bbox=dict(boxstyle="round,pad=0.4", fc=self.clr_panel, ec=self.clr_grid, alpha=0.95),
            color=self.clr_text,
            zorder=101,
            fontsize=9,
        )
        self.annot.set_in_layout(False)
        self.v_line.set_in_layout(False)
        self.h_line.set_in_layout(False)
        self.annot.set_clip_on(True)
        self.annot.get_bbox_patch().set_clip_on(True)
        self.v_line.set_visible(False)
        self.h_line.set_visible(False)
        self.annot.set_visible(False)

    def _clear_cursor(self):
        if self.v_line:
            self.v_line.remove()
        if self.h_line:
            self.h_line.remove()
        if self.annot:
            self.annot.remove()
        self.v_line = self.h_line = self.annot = self.current_ax = None

    def on_mouse_move(self, event):
        if not event.inaxes:
            if self.current_ax:
                self.v_line.set_visible(False)
                self.h_line.set_visible(False)
                self.annot.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        active_axes = [a for a in self.fig.axes if a.bbox.contains(event.x, event.y)]
        if not active_axes:
            return

        primary_ax = active_axes[0]
        self._setup_cursor_elements(primary_ax)
        x_mouse, y_mouse = event.xdata, event.ydata

        closest_x = None
        closest_y = None
        closest_ax = None
        min_y_dist = float("inf")

        for ax in active_axes:
            for line in ax.get_lines():
                x_data = np.asarray(line.get_xdata())
                y_data = np.asarray(line.get_ydata())
                if len(x_data) < 10 or not line.get_visible():
                    continue
                idx = int(np.abs(x_data - x_mouse).argmin())
                x_val = x_data[idx]
                y_val = y_data[idx]
                y_dist = abs(y_val - y_mouse)
                if y_dist < min_y_dist:
                    min_y_dist = y_dist
                    closest_x = x_val
                    closest_y = y_val
                    closest_ax = ax

        if closest_x is None or closest_ax is None:
            return

        x_label = closest_ax.get_xlabel()
        y_label = closest_ax.get_ylabel()
        x_unit = x_label.split("[")[-1].split("]")[0] if "[" in x_label else ""
        y_unit = y_label.split("[")[-1].split("]")[0] if "[" in y_label else ""

        self.v_line.set_xdata([closest_x, closest_x])
        self.h_line.set_ydata([closest_y, closest_y])
        self.annot.xy = (closest_x, closest_y)
        self.annot.set_text(f"X: {closest_x:.2f} {x_unit}\nY: {closest_y:.2f} {y_unit}")
        self.v_line.set_visible(True)
        self.h_line.set_visible(True)
        self.annot.set_visible(True)
        self.fig.canvas.draw_idle()


# =========================================================================
# GUI ONLY: CONFIG COLLECTION
# =========================================================================


def get_configs_from_gui(baseline_path: str | Path = "setup_baseline.json") -> list[SteeringConfig]:
    """Launch a GUI to collect configs. No calculations happen here."""
    baseline_path = Path(baseline_path)
    try:
        with baseline_path.open("r", encoding="utf-8") as f:
            default_cfg_dict = json.load(f)
    except FileNotFoundError:
        messagebox.showerror("Missing File", f"Could not find '{baseline_path}'.")
        return []

    active_configs: list[SteeringConfig] = []

    root = tk.Tk()
    root.title("FSAE Steering Effort Setup")
    root.geometry("600x550")

    frame_left = ttk.LabelFrame(root, text="Manual Override (Defaults to Baseline)")
    frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    entries = {}
    for row, (key, val) in enumerate(default_cfg_dict.items()):
        ttk.Label(frame_left, text=key).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        entry = ttk.Entry(frame_left, width=15)
        entry.insert(0, str(val))
        entry.grid(row=row, column=1, padx=5, pady=2)
        entries[key] = entry

    frame_right = ttk.LabelFrame(root, text="Select Additional Configs to Compare")
    frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    ttk.Label(frame_right, text="Hold CTRL to select multiple files:", font=("Arial", 9, "italic")).pack(padx=5, pady=5)
    listbox = tk.Listbox(frame_right, selectmode=tk.MULTIPLE, width=30)
    json_files = glob.glob("*.json")
    for json_file in json_files:
        if Path(json_file).name != baseline_path.name:
            listbox.insert(tk.END, json_file)
    listbox.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

    def start_calc():
        try:
            manual_cfg_dict = {key: float(entry.get()) for key, entry in entries.items()}
            active_configs.append(SteeringConfig.from_dict(manual_cfg_dict, name="Manual Setup Override"))

            for i in listbox.curselection():
                filename = listbox.get(i)
                active_configs.append(SteeringConfig.from_json(filename, name=filename))
        except (ValueError, KeyError) as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        root.destroy()

    ttk.Button(root, text="Run Calculations", command=start_calc).pack(side=tk.BOTTOM, pady=15)
    root.mainloop()
    return active_configs


# =========================================================================
# APP ORCHESTRATION
# =========================================================================


def run_app(use_gui: bool = True, baseline_path: str | Path = "setup_baseline.json") -> None:
    """Coordinate config collection, pure calculations, reporting, export, and plotting."""
    if use_gui:
        print("Initializing FSAE Steering Effort Tool GUI...")
        configs = get_configs_from_gui(baseline_path)
    else:
        configs = [load_baseline_config(baseline_path)]

    if not configs:
        print("Configuration cancelled. Exiting.")
        return

    persistent_cursors = []

    for cfg in configs:
        result = calculate_steering_effort(cfg)
        sensitivity = run_sensitivity_sweep(cfg)

        print_result_summary(result)
        export_path = export_results_csv(result)
        print(f"\nResults exported to {export_path}")

        fig_results = plot_results(result)
        fig_sens = plot_sensitivity(sensitivity)

        persistent_cursors.append(LineSnapCursor(fig_results))
        persistent_cursors.append(LineSnapCursor(fig_sens))

        print("\n=== RUN COMPLETE ===")

    plt.show()


if __name__ == "__main__":
    # Run `python Effort_Calculator_refactored.py --no-gui` to use only setup_baseline.json.
    run_app(use_gui="--no-gui" not in sys.argv)
