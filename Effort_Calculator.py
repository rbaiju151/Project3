"""
=========================================================================
 FSAE RACK & PINION STATIC STEERING EFFORT PREDICTION TOOL
=========================================================================
 Predicts driver steering effort across full steering travel using a
 physics-based model.

 Units: US Customary internally, converted to SI for output.
=========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================================================================
#  SECTION 1 – VEHICLE INPUT PARAMETERS
# =========================================================================
print("Calculating")
# --- Physical constants ---
G_FPS2    = 32.1741       # [ft/s²]  gravitational acceleration
IN2M      = 0.0254        # [in → m]
LBF2N     = 4.44822       # [lbf → N]
INLB2NM   = 0.112985      # [in·lbf → N·m]
DEG2RAD   = np.pi / 180.0

# --- Tyre / Road ---
MU = 1.63                 # [-] peak tyre–road coefficient of friction

# --- Vehicle mass & distribution ---
W_CAR_LBF      = 588.9                         # [lbf] total vehicle + driver weight
FRONT_DIST     = 0.50                           # [-]   fraction of weight on front axle
W_FRONT_LBF    = W_CAR_LBF * FRONT_DIST        # [lbf] total front-axle load
W_PER_FRONT_LBF = W_FRONT_LBF / 2.0           # [lbf] load per front corner

# --- Steering geometry ---
SCRUB_RADIUS_IN    = 1.595   # [in]  lateral offset of tyre contact patch from KP axis
MECH_TRAIL_IN      = 0.360   # [in]  longitudinal offset of contact patch from KP axis
PNEUM_TRAIL_IN     = 1.000   # [in]  pneumatic trail estimate
CASTER_ANGLE_DEG   = 2.964   # [deg] caster angle (positive → top of KP axis rearward)
KPI_ANGLE_DEG      = 9.130   # [deg] kingpin inclination (positive → top inward)
TOE_LINK_ANGLE_DEG = 1.200   # [deg] toe-link angle from vertical / lateral plane

# --- Steering rack / column ---
PINION_RADIUS_IN     = 0.637  # [in]      effective pinion pitch radius
SW_RADIUS_IN         = 4.724  # [in]      steering wheel radius
TIEROD_MOMENT_ARM_IN = 3.288  # [in]      perpendicular distance: outer tie-rod pickup → KP axis
COL_FRICTION_INLB    = 26.552 # [in·lbf]  steering column + upper-joint friction

# --- KP axis to contact-patch distances ---
KP_PATCH_STATIC_IN  = 1.635  # [in] static position
KP_PATCH_DYNAMIC_IN = 1.917  # [in] at 30° wheel angle

# --- Steering travel ---
MAX_SW_ANGLE_DEG = 180   # [deg] maximum steering-wheel sweep
N_STEPS          = 1000  # number of evaluation points

# =========================================================================
#  SECTION 2 – DERIVED RACK KINEMATICS
# =========================================================================

rack_speed_in_per_rev = 2.0 * np.pi * PINION_RADIUS_IN  # [in/rev]

print("=== FSAE Steering Effort Tool – Derived Parameters ===")
print(f"  Rack speed                   : {rack_speed_in_per_rev:.4f} in/rev")
print(f"  Front corner load            : {W_PER_FRONT_LBF:.2f} lbf")
print(f"  Max lateral tyre force/corner: {MU * W_PER_FRONT_LBF:.2f} lbf  (µ × W)")

# =========================================================================
#  SECTION 3 – SWEEP OVER STEERING WHEEL ANGLE
# =========================================================================

sw_angles_deg = np.linspace(0.0, MAX_SW_ANGLE_DEG, N_STEPS)
sw_angles_rad = sw_angles_deg * DEG2RAD

# Rack displacement from steering wheel rotation
rack_disp_in = sw_angles_rad * PINION_RADIUS_IN

# Wheel steer angle (clipped to avoid arcsin domain error)
wheel_angle_rad = np.arcsin(np.minimum(rack_disp_in / TIEROD_MOMENT_ARM_IN, 0.9999))
wheel_angle_deg = wheel_angle_rad / DEG2RAD

# KP-to-patch distance: linearly interpolated from static → dynamic over 0°→30°,
# then clamped so it never falls below the static value.
kp_patch_in = KP_PATCH_STATIC_IN + (
    (KP_PATCH_DYNAMIC_IN - KP_PATCH_STATIC_IN) * (wheel_angle_deg / 30.0)
)
kp_patch_in = np.maximum(kp_patch_in, KP_PATCH_STATIC_IN)

# Pre-convert geometry angles once (avoids repeated scalar multiplications below)
caster_rad   = CASTER_ANGLE_DEG   * DEG2RAD
kpi_rad      = KPI_ANGLE_DEG      * DEG2RAD
toe_link_rad = TOE_LINK_ANGLE_DEG * DEG2RAD

# =========================================================================
#  SECTION 4 – ALIGNING MOMENT AT CONTACT PATCH
# =========================================================================

total_trail_in = MECH_TRAIL_IN + PNEUM_TRAIL_IN

T_trail_inlb = MU * W_PER_FRONT_LBF * total_trail_in    * np.cos(wheel_angle_rad)
T_scrub_inlb = MU * W_PER_FRONT_LBF * SCRUB_RADIUS_IN   * np.sin(wheel_angle_rad)

T_align_inlb = T_trail_inlb + T_scrub_inlb

# =========================================================================
#  SECTION 5 – KINGPIN GEOMETRY RESTORATION TORQUE
# =========================================================================

T_caster_inlb = (
    W_PER_FRONT_LBF * MECH_TRAIL_IN * np.sin(wheel_angle_rad) * np.sin(caster_rad)
)
T_kpi_inlb = (
    W_PER_FRONT_LBF * SCRUB_RADIUS_IN
    * np.sin(wheel_angle_rad) * np.sin(kpi_rad) * np.cos(caster_rad)
)

T_geom_inlb = T_caster_inlb + T_kpi_inlb

# =========================================================================
#  SECTION 6 – TOE-LINK ANGLE AMPLIFICATION FACTOR
# =========================================================================

toe_link_factor = 1.0 / np.cos(toe_link_rad)

# =========================================================================
#  SECTION 7 – REFER TORQUES TO RACK AND STEERING WHEEL
# =========================================================================

T_kp_total_inlb  = T_align_inlb + T_geom_inlb

tierod_arm_eff_in = TIEROD_MOMENT_ARM_IN * np.cos(wheel_angle_rad)
F_tierod_lbf      = T_kp_total_inlb / tierod_arm_eff_in

F_rack_lbf    = F_tierod_lbf * toe_link_factor
T_pinion_inlb = F_rack_lbf * PINION_RADIUS_IN

# =========================================================================
#  SECTION 8 – COLUMN FRICTION (constant offset)
# =========================================================================

# COL_FRICTION_INLB is already defined as a constant; no further derivation needed.

# =========================================================================
#  SECTION 9 – TOTAL STEERING EFFORT
# =========================================================================

T_total_inlb = T_pinion_inlb + COL_FRICTION_INLB

# Convert all torques and forces to SI once
T_align_Nm  = T_align_inlb            * INLB2NM
T_geom_Nm   = T_geom_inlb             * INLB2NM
T_pinion_Nm = T_pinion_inlb           * INLB2NM
T_col_Nm    = COL_FRICTION_INLB       * INLB2NM   # scalar – broadcast where needed
T_total_Nm  = T_total_inlb            * INLB2NM
T_caster_Nm = T_caster_inlb           * INLB2NM
T_kpi_Nm    = T_kpi_inlb              * INLB2NM
T_trail_Nm  = T_trail_inlb            * INLB2NM
T_scrub_Nm  = T_scrub_inlb            * INLB2NM
F_rack_N    = F_rack_lbf              * LBF2N

# =========================================================================
#  SECTION 10 – SUMMARY TABLE AT KEY STEER ANGLES
# =========================================================================

print("\n=== Steering Effort Summary at Key Wheel Angles ===")
header = (
    f"{'SW [deg]':<12} {'Wheel [deg]':<12} {'T_align [Nm]':<14}"
    f"{'T_geom [Nm]':<13} {'T_col [Nm]':<12} {'T_total [Nm]':<13} {'F_rack [N]':<10}"
)
print(header)
print("-" * len(header))

check_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 150, 180]
for ca in check_angles:
    idx = np.argmin(np.abs(sw_angles_deg - ca))
    print(
        f"{sw_angles_deg[idx]:<12.1f} {wheel_angle_deg[idx]:<12.2f}"
        f"{T_align_Nm[idx]:<14.3f} {T_geom_Nm[idx]:<13.3f}"
        f"{T_col_Nm:<12.3f} {T_total_Nm[idx]:<13.3f} {F_rack_N[idx]:<10.2f}"
    )

# =========================================================================
#  SECTION 11 – COMPONENT BREAKDOWN AT ~30° WHEEL ANGLE
# =========================================================================

idx_30 = np.argmin(np.abs(wheel_angle_deg - 30.0))
print(f"\n=== Component Breakdown at ~30° Wheel Angle ===")
print(f"  SW angle at 30° wheel  : {sw_angles_deg[idx_30]:.1f} deg")
print(f"  Trail torque           : {T_trail_Nm[idx_30]:+.3f} N·m")
print(f"  Scrub torque           : {T_scrub_Nm[idx_30]:+.3f} N·m")
print(f"  Caster restoration     : {T_caster_Nm[idx_30]:+.3f} N·m")
print(f"  KPI restoration        : {T_kpi_Nm[idx_30]:+.3f} N·m")
print(f"  Sub-total at KP axis   : {(T_align_Nm[idx_30] + T_geom_Nm[idx_30]):+.3f} N·m")
print(f"  Referred to pinion     : {T_pinion_Nm[idx_30]:+.3f} N·m")
print(f"  Column friction        : {T_col_Nm:+.3f} N·m")
print(f"  TOTAL steering effort  : {T_total_Nm[idx_30]:+.3f} N·m")

# =========================================================================
#  SECTION 12 – PLOTTING
# =========================================================================

# ── Colour palette (dark theme) ────────────────────────────────────────────
clr_bg    = (0.08, 0.08, 0.12)
clr_panel = (0.12, 0.12, 0.18)
clr_total = (0.20, 0.85, 0.50)
clr_align = (0.95, 0.60, 0.10)
clr_geom  = (0.30, 0.65, 1.00)
clr_col   = (0.85, 0.30, 0.30)
clr_grid  = (0.25, 0.25, 0.30)
clr_text  = (0.92, 0.92, 0.95)

plt.rcParams.update({
    "text.color":        clr_text,
    "axes.labelcolor":   clr_text,
    "xtick.color":       clr_text,
    "ytick.color":       clr_text,
})


def style_ax(ax: plt.Axes) -> None:
    """Apply the dark-theme style to a single Axes object."""
    ax.set_facecolor(clr_panel)
    ax.grid(True, color=clr_grid, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_color(clr_text)


fig = plt.figure(figsize=(14, 9), facecolor=clr_bg)
fig.canvas.manager.set_window_title("FSAE Steering Effort Prediction")

# ── Plot 1: Total effort vs SW angle ──────────────────────────────────────
ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
style_ax(ax1)

ax1.fill_between([0, MAX_SW_ANGLE_DEG], 6, 15,
                 color=clr_total, alpha=0.08, edgecolor="none")
ax1.axhline(6,  color=(0.50, 0.95, 0.65), linestyle="--",
            linewidth=1, alpha=0.6, label="6 N·m target min")
ax1.axhline(15, color=(0.50, 0.95, 0.65), linestyle="--",
            linewidth=1, alpha=0.6, label="15 N·m target max")

ax1.plot(sw_angles_deg, T_col_Nm * np.ones_like(sw_angles_deg),
         "-",  color=clr_col,   linewidth=1.2, label="Column Friction")
ax1.plot(sw_angles_deg, T_align_Nm + T_col_Nm,
         "-",  color=clr_align, linewidth=1.5, label="+ Aligning Moment")
ax1.plot(sw_angles_deg, T_total_Nm,
         "-",  color=clr_total, linewidth=2.5, label="Total Driver Effort")

ax1.axvline(105, color=clr_text, linestyle="--", alpha=0.5,
            linewidth=1, label="105° SW limit")

ax1.set_xlabel("Steering Wheel Angle [deg]", fontsize=11)
ax1.set_ylabel("Torque [N·m]",               fontsize=11)
ax1.set_title("Total Steering Effort vs. Steering Wheel Angle",
              fontsize=13, fontweight="bold")
ax1.legend(loc="upper left", facecolor=clr_panel, edgecolor=clr_grid,
           fontsize=9, labelcolor=clr_text)
ax1.set_xlim([0, MAX_SW_ANGLE_DEG])
ax1.set_ylim([0, np.max(T_total_Nm) * 1.1])

# ── Plot 2: Stacked component breakdown ───────────────────────────────────
ax2 = plt.subplot2grid((2, 3), (0, 2))
style_ax(ax2)

y1 = T_col_Nm * np.ones_like(sw_angles_deg)
y2 = T_trail_Nm
y3 = T_scrub_Nm
y4 = T_caster_Nm
y5 = T_kpi_Nm

colors_stack = [
    clr_col,
    (0.95, 0.80, 0.20),
    (0.85, 0.45, 0.10),
    (0.30, 0.55, 1.00),
    (0.60, 0.30, 1.00),
]
labels_stack = [
    "Column Friction",
    "Trail Aligning",
    "Scrub Aligning",
    "Caster Restore",
    "KPI Restore",
]

ax2.stackplot(sw_angles_deg, y1, y2, y3, y4, y5,
              labels=labels_stack, colors=colors_stack, alpha=0.75, edgecolor="none")
ax2.plot(sw_angles_deg, T_total_Nm, "-", color=clr_total, linewidth=2, label="Total")

ax2.set_xlabel("Steering Wheel Angle [deg]", fontsize=10)
ax2.set_ylabel("Torque [N·m]",               fontsize=10)
ax2.set_title("Stacked Component Breakdown",  fontsize=11, fontweight="bold")
ax2.legend(loc="upper left", facecolor=clr_panel, edgecolor=clr_grid,
           fontsize=7, labelcolor=clr_text)
ax2.set_xlim([0, MAX_SW_ANGLE_DEG])
ax2.set_ylim([0, np.max(T_total_Nm) * 1.15])

# ── Plot 3: Wheel steer angle vs SW angle ─────────────────────────────────
ax3 = plt.subplot2grid((2, 3), (1, 0))
style_ax(ax3)
ax3.plot(sw_angles_deg, wheel_angle_deg, "-", color=(0.80, 0.70, 1.00), linewidth=2)
ax3.set_xlabel("Steering Wheel Angle [deg]", fontsize=10)
ax3.set_ylabel("Wheel Steer Angle [deg]",    fontsize=10)
ax3.set_title("Wheel Angle vs. SW Angle",    fontsize=11, fontweight="bold")
ax3.set_xlim([0, MAX_SW_ANGLE_DEG])

# ── Plot 4: Rack force vs SW angle ────────────────────────────────────────
ax4 = plt.subplot2grid((2, 3), (1, 1))
style_ax(ax4)
ax4.plot(sw_angles_deg, F_rack_N, "-", color=(1.0, 0.60, 0.20), linewidth=2)
ax4.set_xlabel("Steering Wheel Angle [deg]", fontsize=10)
ax4.set_ylabel("Rack Force [N]",             fontsize=10)
ax4.set_title("Required Rack Force",         fontsize=11, fontweight="bold")
ax4.set_xlim([0, MAX_SW_ANGLE_DEG])

# ── Plot 5: KP-to-patch distance and effective tie-rod arm ────────────────
ax5 = plt.subplot2grid((2, 3), (1, 2))
style_ax(ax5)

l1 = ax5.plot(wheel_angle_deg, kp_patch_in * IN2M * 1000,
              "-", color=(0.40, 0.90, 0.90), linewidth=2, label="KP–Patch Distance")
ax5.set_ylabel("KP–Patch Dist [mm]", color=(0.40, 0.90, 0.90), fontsize=10)
ax5.set_xlabel("Wheel Steer Angle [deg]",    fontsize=10)
ax5.tick_params(axis="y", colors=(0.40, 0.90, 0.90))

ax5_rt = ax5.twinx()
l2 = ax5_rt.plot(wheel_angle_deg, tierod_arm_eff_in * IN2M * 1000,
                 "--", color=(1.0, 0.70, 0.40), linewidth=2, label="Effective Tie-rod Arm")
ax5_rt.set_ylabel("Effective Tie-rod Arm [mm]", color=(1.0, 0.70, 0.40), fontsize=10)
ax5_rt.tick_params(axis="y", colors=(1.0, 0.70, 0.40))
for spine in ax5_rt.spines.values():
    spine.set_visible(False)

ax5.set_title("Geometry Arms vs. Wheel Angle", fontsize=11, fontweight="bold")
combined_lines  = l1 + l2
combined_labels = [line.get_label() for line in combined_lines]
ax5.legend(combined_lines, combined_labels, loc="lower right",
           facecolor=clr_panel, edgecolor=clr_grid, fontsize=8, labelcolor=clr_text)

# ── Super title ──────────────────────────────────────────────────────────
fig.suptitle(
    f"FSAE Rack & Pinion Steering Effort Prediction Tool\n"
    f"µ={MU} | W_front={W_PER_FRONT_LBF:.0f} lbf/corner"
    f" | Caster={CASTER_ANGLE_DEG}° | KPI={KPI_ANGLE_DEG}°",
    fontsize=14, fontweight="bold", color=clr_text,
)

plt.tight_layout()

# =========================================================================
#  SECTION 13 – SENSITIVITY SWEEP
# =========================================================================

fig2, axs = plt.subplots(2, 3, figsize=(12, 7), facecolor=clr_bg)
fig2.canvas.manager.set_window_title("Sensitivity Analysis")

param_labels = [
    "Caster Angle [deg]",
    "KPI Angle [deg]",
    "Scrub Radius [in]",
    "Mechanical Trail [in]",
    "Tie-rod Moment Arm [in]",
    "Coefficient of Friction",
]
base_vals = [
    CASTER_ANGLE_DEG,
    KPI_ANGLE_DEG,
    SCRUB_RADIUS_IN,
    MECH_TRAIL_IN,
    TIEROD_MOMENT_ARM_IN,
    MU,
]
DELTA_PCT = 0.25  # ±25 % variation

# Index corresponding to 105° SW (evaluation reference point)
ref_idx = int(np.round((N_STEPS - 1) * 105.0 / MAX_SW_ANGLE_DEG))

colors_sens = plt.cm.tab10(np.linspace(0, 1, 10))

for sp, ax_s in enumerate(axs.flatten()):
    style_ax(ax_s)

    sweep_vals = np.linspace(
        base_vals[sp] * (1 - DELTA_PCT),
        base_vals[sp] * (1 + DELTA_PCT),
        20,
    )
    efforts = np.zeros_like(sweep_vals)

    for sv_idx, sv_val in enumerate(sweep_vals):
        p = list(base_vals)
        p[sp] = sv_val

        cas_r  = p[0] * DEG2RAD
        kpi_r  = p[1] * DEG2RAD
        scr    = p[2]
        mtr    = p[3]
        tarm   = p[4]
        mu_s   = p[5]

        wa_r     = np.arcsin(min(rack_disp_in[ref_idx] / tarm, 0.9999))
        T_a      = (mu_s * W_PER_FRONT_LBF * (mtr + PNEUM_TRAIL_IN) * np.cos(wa_r)
                    + mu_s * W_PER_FRONT_LBF * scr * np.sin(wa_r))
        T_g      = (W_PER_FRONT_LBF * mtr  * np.sin(wa_r) * np.sin(cas_r)
                    + W_PER_FRONT_LBF * scr * np.sin(wa_r) * np.sin(kpi_r) * np.cos(cas_r))
        tarm_eff = tarm * np.cos(wa_r)
        F_r      = (T_a + T_g) / tarm_eff / np.cos(toe_link_rad)
        efforts[sv_idx] = (F_r * PINION_RADIUS_IN + COL_FRICTION_INLB) * INLB2NM

    xn = sweep_vals / base_vals[sp]  # normalised x-axis (1.0 = baseline)
    ax_s.plot(xn, efforts, "-o",
              color=colors_sens[sp], linewidth=2,
              markersize=4, markerfacecolor=colors_sens[sp])

    ax_s.axvline(1.0, color=clr_text, linestyle="--", alpha=0.5, linewidth=1)
    ax_s.axhline(6,   color=(0.50, 0.95, 0.65), linestyle="--", linewidth=0.8, alpha=0.5)
    ax_s.axhline(15,  color=(0.50, 0.95, 0.65), linestyle="--", linewidth=0.8, alpha=0.5)

    ax_s.set_xlabel("Parameter Multiple of Baseline", fontsize=9)
    ax_s.set_ylabel("Effort @ 105° SW [N·m]",         fontsize=9)
    ax_s.set_title(param_labels[sp], fontsize=10, fontweight="bold")

fig2.suptitle(
    "Sensitivity Analysis – Effort at 105° Steering Wheel Angle",
    fontsize=13, fontweight="bold", color=clr_text,
)
plt.tight_layout()

# =========================================================================
#  SECTION 14 – EXPORT RESULTS TO CSV
# =========================================================================

export_data = {
    "SW_Angle_deg":       sw_angles_deg,
    "Wheel_Angle_deg":    wheel_angle_deg,
    "T_Trail_Nm":         T_trail_Nm,
    "T_Scrub_Nm":         T_scrub_Nm,
    "T_Caster_Nm":        T_caster_Nm,
    "T_KPI_Nm":           T_kpi_Nm,
    "T_Align_Nm":         T_align_Nm,
    "T_Geom_Nm":          T_geom_Nm,
    "T_Pinion_Nm":        T_pinion_Nm,
    "T_ColFriction_Nm":   T_col_Nm * np.ones(N_STEPS),
    "T_Total_Nm":         T_total_Nm,
    "F_Rack_N":           F_rack_N,
}

df_export = pd.DataFrame(export_data)
df_export.to_csv("steering_effort_results.csv", index=False)
print("\n✓ Results exported to steering_effort_results.csv")

# =========================================================================
#  SECTION 15 – FINAL VALIDATION CHECK
# =========================================================================

TARGET_MIN_NM = 6.0
TARGET_MAX_NM = 15.0

i0   = np.argmin(np.abs(sw_angles_deg - 0.0))
i105 = np.argmin(np.abs(sw_angles_deg - 105.0))

print("\n=== TARGET VALIDATION ===")
for label, idx in [("  0° SW", i0), ("105° SW", i105)]:
    effort = T_total_Nm[idx]
    status = (
        f"[✓ Within {TARGET_MIN_NM}–{TARGET_MAX_NM} N·m target]"
        if TARGET_MIN_NM <= effort <= TARGET_MAX_NM
        else f"[! Outside {TARGET_MIN_NM}–{TARGET_MAX_NM} N·m target]"
    )
    print(f"  Effort at {label} : {effort:.3f} N·m  {status}")

print("\n=== RUN COMPLETE ===")

plt.show()

# END OF FILE