"""
=========================================================================
 frontend.py
 FSAE RACK & PINION STATIC STEERING EFFORT PREDICTION TOOL
 
 Handles all Tkinter GUI interactions, plotting via Matplotlib, and app
 orchestration. Relies on backend.py for calculations.
=========================================================================
"""

import sys
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox

# --- Import from our newly created backend ---
from backend import (
    SteeringConfig,
    SteeringResults,
    calculate_steering_effort,
    run_sensitivity_sweep,
    print_result_summary,
    export_results_csv,
    load_baseline_config,
    TARGET_MIN_NM,
    TARGET_MAX_NM,
    IN2M
)

# =========================================================================
# PLOTTING CONSTANTS & THEMING
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
    plt.rcParams.update({
        "text.color": CLR_TEXT,
        "axes.labelcolor": CLR_TEXT,
        "xtick.color": CLR_TEXT,
        "ytick.color": CLR_TEXT,
    })

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
    ax1.axhline(TARGET_MIN_NM, color=(0.50, 0.95, 0.65), linestyle="--", linewidth=1, alpha=0.6, label=f"{TARGET_MIN_NM} N*m target min")
    ax1.axhline(TARGET_MAX_NM, color=(0.50, 0.95, 0.65), linestyle="--", linewidth=1, alpha=0.6, label=f"{TARGET_MAX_NM} N*m target max")
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
        fontsize=14, fontweight="bold", color=CLR_TEXT,
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
            df["parameter_multiple_of_baseline"], df["effort_Nm"], "-o",
            color=colors[sp], linewidth=2, markersize=4, markerfacecolor=colors[sp],
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
        for ax in self.fig.axes:
            ax.set_autoscale_on(False)
            ax.set_xlim(ax.get_xlim())
            ax.set_ylim(ax.get_ylim())
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

    def _setup_cursor_elements(self, ax):
        if self.current_ax == ax: return
        self._clear_cursor()
        self.current_ax = ax
        self.v_line = ax.axvline(color=self.clr_text, linestyle=":", alpha=0.6, zorder=100)
        self.h_line = ax.axhline(color=self.clr_text, linestyle=":", alpha=0.6, zorder=100)
        self.annot = ax.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points", annotation_clip=True,
            bbox=dict(boxstyle="round,pad=0.4", fc=self.clr_panel, ec=self.clr_grid, alpha=0.95),
            color=self.clr_text, zorder=101, fontsize=9,
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
        if self.v_line: self.v_line.remove()
        if self.h_line: self.h_line.remove()
        if self.annot: self.annot.remove()
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
        if not active_axes: return

        primary_ax = active_axes[0]
        self._setup_cursor_elements(primary_ax)
        x_mouse, y_mouse = event.xdata, event.ydata
        closest_x, closest_y, closest_ax = None, None, None
        min_y_dist = float("inf")

        for ax in active_axes:
            for line in ax.get_lines():
                x_data = np.asarray(line.get_xdata())
                y_data = np.asarray(line.get_ydata())
                if len(x_data) < 10 or not line.get_visible(): continue
                idx = int(np.abs(x_data - x_mouse).argmin())
                x_val, y_val = x_data[idx], y_data[idx]
                y_dist = abs(y_val - y_mouse)
                if y_dist < min_y_dist:
                    min_y_dist = y_dist
                    closest_x, closest_y, closest_ax = x_val, y_val, ax

        if closest_x is None or closest_ax is None: return

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
        # Call to the backend logic
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
    # Run `python frontend.py --no-gui` to use only setup_baseline.json.
    run_app(use_gui="--no-gui" not in sys.argv)