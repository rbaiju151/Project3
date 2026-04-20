"""
=========================================================================
 FSAE RACK & PINION STATIC STEERING EFFORT PREDICTION TOOL
=========================================================================
 Predicts driver steering effort across full steering travel using a
 physics-based model.
 
 Units   : US Customary internally, converted to SI for output
=========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================================================================
#  SECTION 1 – VEHICLE INPUT PARAMETERS
# =========================================================================

# --- Physical constants ---
g_fps2       = 32.1741           # [ft/s²] gravitational acceleration
in2m         = 0.0254            # [in -> m]
lbf2N        = 4.44822           # [lbf -> N]
inlb2Nm      = 0.112985          # [in·lbf -> N·m]
deg2rad      = np.pi / 180

# --- Tyre / Road ---
mu           = 1.63              # [-] peak tyre-road coefficient of friction

# --- Vehicle mass & distribution ---
W_car_lbf    = 588.9             # [lbf] total vehicle + driver weight
front_dist   = 0.50              # [-] fraction of weight on front axle
W_front_lbf  = W_car_lbf * front_dist  # [lbf] total front axle load
W_per_front_lbf = W_front_lbf / 2      # [lbf] load per front corner

# --- Steering geometry (given) ---
scrub_radius_in      = 1.595     # [in] lateral offset of tyre contact patch from KP axis
mech_trail_in        = 0.360     # [in] longitudinal offset of contact patch from KP axis
pneum_trail_in       = 1.000     # [in] pneumatic trail estimate
caster_angle_deg     = 2.964     # [deg] caster angle (positive = top of KP axis rearward)
kpi_angle_deg        = 9.130     # [deg] kingpin inclination angle (positive = top inward)
toe_link_angle_deg   = 1.200     # [deg] angle of toe links from vertical / lateral plane

# --- Steering rack / column ---
pinion_radius_in     = 0.637     # [in] effective pinion pitch radius
sw_radius_in         = 4.724     # [in] steering wheel radius
tierod_moment_arm_in = 3.288     # [in] perpendicular distance from outer tie-rod pickup to KP axis
col_friction_inlb    = 26.552    # [in·lbf] steering column + upper friction

# --- KP axis to contact patch distances ---
kp_patch_static_in   = 1.635     # [in] KP axis to contact patch – static
kp_patch_dynamic_in  = 1.917     # [in] KP axis to contact patch – dynamic (at 30° wheel angle)

# --- Steering travel ---
max_sw_angle_deg = 180           # [deg] maximum steering wheel sweep
n_steps          = 1000          # number of evaluation points

# =========================================================================
#  SECTION 2 – DERIVED RACK KINEMATICS
# =========================================================================

rack_speed_in_per_rev = 2 * np.pi * pinion_radius_in   # [in/rev]

print('=== FSAE Steering Effort Tool – Derived Parameters ===')
print(f'Rack speed (in/rev)         : {rack_speed_in_per_rev:.4f} in/rev')
print(f'Front corner load           : {W_per_front_lbf:.2f} lbf')
print(f'Max lateral tyre force/corner: {mu * W_per_front_lbf:.2f} lbf (µ × W)')

# =========================================================================
#  SECTION 3 – SWEEP OVER STEERING WHEEL ANGLE
# =========================================================================

sw_angles_deg  = np.linspace(0, max_sw_angle_deg, n_steps)
sw_angles_rad  = sw_angles_deg * deg2rad

rack_disp_in   = sw_angles_rad * pinion_radius_in          

# Wheel steer angle
wheel_angle_rad = np.arcsin(np.minimum(rack_disp_in / tierod_moment_arm_in, 0.9999))
wheel_angle_deg = wheel_angle_rad / deg2rad

# Interpolate KP-to-patch distance linearly 0->30°
kp_patch_in = kp_patch_static_in + (kp_patch_dynamic_in - kp_patch_static_in) * (wheel_angle_deg / 30)
kp_patch_in = np.maximum(kp_patch_in, kp_patch_static_in)

# Convert key angles to radians
caster_rad   = caster_angle_deg * deg2rad
kpi_rad      = kpi_angle_deg    * deg2rad
toe_link_rad = toe_link_angle_deg * deg2rad

# =========================================================================
#  SECTION 4 – ALIGNING MOMENT AT CONTACT PATCH
# =========================================================================

total_trail_in = mech_trail_in + pneum_trail_in    

T_trail_inlb = mu * W_per_front_lbf * total_trail_in * np.cos(wheel_angle_rad)
T_scrub_inlb = mu * W_per_front_lbf * scrub_radius_in * np.sin(wheel_angle_rad)

T_align_inlb = T_trail_inlb + T_scrub_inlb    

# =========================================================================
#  SECTION 5 – KINGPIN GEOMETRY RESTORATION TORQUE
# =========================================================================

T_caster_inlb = W_per_front_lbf * mech_trail_in * np.sin(wheel_angle_rad) * np.sin(caster_rad)
T_kpi_inlb    = W_per_front_lbf * scrub_radius_in * np.sin(wheel_angle_rad) * np.sin(kpi_rad) * np.cos(caster_rad)

T_geom_inlb   = T_caster_inlb + T_kpi_inlb

# =========================================================================
#  SECTION 6 – TOE LINK ANGLE AMPLIFICATION
# =========================================================================

toe_link_factor = 1 / np.cos(toe_link_rad)

# =========================================================================
#  SECTION 7 – REFER TORQUES BACK THROUGH TIE ROD TO RACK TO STEERING WHEEL
# =========================================================================

T_kp_total_inlb = T_align_inlb + T_geom_inlb

tierod_arm_eff_in = tierod_moment_arm_in * np.cos(wheel_angle_rad)
F_tierod_lbf      = T_kp_total_inlb / tierod_arm_eff_in

F_rack_lbf        = F_tierod_lbf * toe_link_factor
T_pinion_inlb     = F_rack_lbf * pinion_radius_in

# =========================================================================
#  SECTION 8 – COLUMN FRICTION
# =========================================================================

T_col_friction_inlb = col_friction_inlb

# =========================================================================
#  SECTION 9 – TOTAL STEERING EFFORT
# =========================================================================

T_total_inlb = T_pinion_inlb + T_col_friction_inlb

T_align_Nm   = T_align_inlb   * inlb2Nm
T_geom_Nm    = T_geom_inlb    * inlb2Nm
T_pinion_Nm  = T_pinion_inlb  * inlb2Nm
T_col_Nm     = T_col_friction_inlb * inlb2Nm
T_total_Nm   = T_total_inlb   * inlb2Nm
T_caster_Nm  = T_caster_inlb  * inlb2Nm
T_kpi_Nm     = T_kpi_inlb     * inlb2Nm
T_trail_Nm   = T_trail_inlb   * inlb2Nm
T_scrub_Nm   = T_scrub_inlb   * inlb2Nm
F_rack_N     = F_rack_lbf     * lbf2N

# =========================================================================
#  SECTION 10 – PRINT SUMMARY TABLE AT KEY STEER ANGLES
# =========================================================================

print('\n=== Steering Effort Summary at Key Wheel Angles ===')
print(f'{"SW [deg]":<12} {"Wheel[deg]":<12} {"T_align[Nm]":<12} {"T_geom[Nm]":<12} {"T_col[Nm]":<12} {"T_total[Nm]":<12} {"F_rack[N]":<12}')
print('-' * 86)

check_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 150, 180]
for ca in check_angles:
    idx = np.argmin(np.abs(sw_angles_deg - ca))
    print(f'{sw_angles_deg[idx]:<12.1f} {wheel_angle_deg[idx]:<12.2f} {T_align_Nm[idx]:<12.3f} {T_geom_Nm[idx]:<12.3f} {T_col_Nm:<12.3f} {T_total_Nm[idx]:<12.3f} {F_rack_N[idx]:<12.2f}')

# =========================================================================
#  SECTION 11 – SENSITIVITY ANALYSIS (CONSOLE)
# =========================================================================

idx_30 = np.argmin(np.abs(wheel_angle_deg - 30))
print('\n=== Component Breakdown at ~30° Wheel Angle ===')
print(f'  SW angle at 30° wheel  : {sw_angles_deg[idx_30]:.1f} deg')
print(f'  Trail torque           : {T_trail_Nm[idx_30]:+.3f} N·m')
print(f'  Scrub torque           : {T_scrub_Nm[idx_30]:+.3f} N·m')
print(f'  Caster restoration     : {T_caster_Nm[idx_30]:+.3f} N·m')
print(f'  KPI restoration        : {T_kpi_Nm[idx_30]:+.3f} N·m')
print(f'  Sub-total at KP axis   : {(T_align_Nm[idx_30]+T_geom_Nm[idx_30]):+.3f} N·m')
print(f'  Referred to pinion     : {T_pinion_Nm[idx_30]:+.3f} N·m')
print(f'  Column friction        : {T_col_Nm:+.3f} N·m')
print(f'  TOTAL steering effort  : {T_total_Nm[idx_30]:+.3f} N·m')

# =========================================================================
#  SECTION 12 – PLOTTING
# =========================================================================

# Custom dark theme colors
clr_bg     = (0.08, 0.08, 0.12)
clr_panel  = (0.12, 0.12, 0.18)
clr_total  = (0.20, 0.85, 0.50)
clr_align  = (0.95, 0.60, 0.10)
clr_geom   = (0.30, 0.65, 1.00)
clr_col    = (0.85, 0.30, 0.30)
clr_grid   = (0.25, 0.25, 0.30)
clr_text   = (0.92, 0.92, 0.95)

plt.rcParams.update({'text.color': clr_text, 'axes.labelcolor': clr_text, 'xtick.color': clr_text, 'ytick.color': clr_text})

fig = plt.figure(figsize=(14, 9), facecolor=clr_bg)
fig.canvas.manager.set_window_title('FSAE Steering Effort Prediction')

# Helper function to style axes
def style_ax(ax):
    ax.set_facecolor(clr_panel)
    ax.grid(True, color=clr_grid, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_color(clr_text)

# ── Plot 1: Total steering effort vs SW angle ──────────────────────────────
ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
style_ax(ax1)
ax1.fill_between([0, max_sw_angle_deg], 6, 15, color=clr_total, alpha=0.08, edgecolor='none')
ax1.axhline(6, color=(0.50, 0.95, 0.65), linestyle='--', linewidth=1, alpha=0.6, label='6 N·m target min')
ax1.axhline(15, color=(0.50, 0.95, 0.65), linestyle='--', linewidth=1, alpha=0.6, label='15 N·m target max')

ax1.plot(sw_angles_deg, T_col_Nm * np.ones_like(sw_angles_deg), '-', color=clr_col, linewidth=1.2, label='Column Friction')
ax1.plot(sw_angles_deg, T_align_Nm + T_col_Nm, '-', color=clr_align, linewidth=1.5, label='+ Aligning Moment')
ax1.plot(sw_angles_deg, T_total_Nm, '-', color=clr_total, linewidth=2.5, label='Total Driver Effort')

ax1.axvline(105, color=clr_text, linestyle='--', alpha=0.5, linewidth=1, label='105° SW limit')

ax1.set_xlabel('Steering Wheel Angle [deg]', fontsize=11)
ax1.set_ylabel('Torque [N·m]', fontsize=11)
ax1.set_title('Total Steering Effort vs. Steering Wheel Angle', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', facecolor=clr_panel, edgecolor=clr_grid, fontsize=9, labelcolor=clr_text)
ax1.set_xlim([0, max_sw_angle_deg])
ax1.set_ylim([0, np.max(T_total_Nm)*1.1])

# ── Plot 2: Component breakdown stacked ────────────────────────────────────
ax2 = plt.subplot2grid((2, 3), (0, 2))
style_ax(ax2)

y1 = T_col_Nm * np.ones_like(sw_angles_deg)
y2 = T_trail_Nm
y3 = T_scrub_Nm
y4 = T_caster_Nm
y5 = T_kpi_Nm

colors_stack = [clr_col, (0.95, 0.80, 0.20), (0.85, 0.45, 0.10), (0.30, 0.55, 1.0), (0.60, 0.30, 1.0)]
labels_stack = ['Column Friction', 'Trail Aligning', 'Scrub Aligning', 'Caster Restore', 'KPI Restore']

ax2.stackplot(sw_angles_deg, y1, y2, y3, y4, y5, labels=labels_stack, colors=colors_stack, alpha=0.75, edgecolor='none')
ax2.plot(sw_angles_deg, T_total_Nm, '-', color=clr_total, linewidth=2, label='Total')

ax2.set_xlabel('Steering Wheel Angle [deg]', fontsize=10)
ax2.set_ylabel('Torque [N·m]', fontsize=10)
ax2.set_title('Stacked Component Breakdown', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', facecolor=clr_panel, edgecolor=clr_grid, fontsize=7, labelcolor=clr_text)
ax2.set_xlim([0, max_sw_angle_deg])
ax2.set_ylim([0, np.max(T_total_Nm)*1.15])

# ── Plot 3: Wheel steer angle vs SW angle ──────────────────────────────────
ax3 = plt.subplot2grid((2, 3), (1, 0))
style_ax(ax3)
ax3.plot(sw_angles_deg, wheel_angle_deg, '-', color=(0.80, 0.70, 1.00), linewidth=2)
ax3.set_xlabel('Steering Wheel Angle [deg]', fontsize=10)
ax3.set_ylabel('Wheel Steer Angle [deg]', fontsize=10)
ax3.set_title('Wheel Angle vs. SW Angle', fontsize=11, fontweight='bold')
ax3.set_xlim([0, max_sw_angle_deg])

# ── Plot 4: Rack force vs SW angle ─────────────────────────────────────────
ax4 = plt.subplot2grid((2, 3), (1, 1))
style_ax(ax4)
ax4.plot(sw_angles_deg, F_rack_N, '-', color=(1.0, 0.60, 0.20), linewidth=2)
ax4.set_xlabel('Steering Wheel Angle [deg]', fontsize=10)
ax4.set_ylabel('Rack Force [N]', fontsize=10)
ax4.set_title('Required Rack Force', fontsize=11, fontweight='bold')
ax4.set_xlim([0, max_sw_angle_deg])

# ── Plot 5: KP-to-patch moment arm vs wheel angle ─────────────────────────
ax5 = plt.subplot2grid((2, 3), (1, 2))
style_ax(ax5)

l1 = ax5.plot(wheel_angle_deg, kp_patch_in * in2m * 1000, '-', color=(0.40, 0.90, 0.90), linewidth=2, label='KP–Patch Distance')
ax5.set_ylabel('KP–Patch Dist [mm]', color=(0.40, 0.90, 0.90), fontsize=10)
ax5.set_xlabel('Wheel Steer Angle [deg]', fontsize=10)
ax5.tick_params(axis='y', colors=(0.40, 0.90, 0.90))

ax5_rt = ax5.twinx()
l2 = ax5_rt.plot(wheel_angle_deg, tierod_arm_eff_in * in2m * 1000, '--', color=(1.0, 0.70, 0.40), linewidth=2, label='Effective Tie-rod Arm')
ax5_rt.set_ylabel('Effective Tie-rod Arm [mm]', color=(1.0, 0.70, 0.40), fontsize=10)
ax5_rt.tick_params(axis='y', colors=(1.0, 0.70, 0.40))
for spine in ax5_rt.spines.values():
    spine.set_visible(False)

ax5.set_title('Geometry Arms vs. Wheel Angle', fontsize=11, fontweight='bold')
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax5.legend(lines, labels, loc='lower right', facecolor=clr_panel, edgecolor=clr_grid, fontsize=8, labelcolor=clr_text)

# ── Super title ─────────────────────────────────────────────────────────────
fig.suptitle(f'FSAE Rack & Pinion Steering Effort Prediction Tool\n26x | µ={mu} | W_{{front}}={W_per_front_lbf:.0f} lbf/corner | Caster={caster_angle_deg}° | KPI={kpi_angle_deg}°', 
             fontsize=14, fontweight='bold', color=clr_text)

plt.tight_layout()

# =========================================================================
#  SECTION 13 – SENSITIVITY SWEEP
# =========================================================================

fig2, axs = plt.subplots(2, 3, figsize=(12, 7), facecolor=clr_bg)
fig2.canvas.manager.set_window_title('Sensitivity Analysis')

params = ['Caster Angle [deg]', 'KPI Angle [deg]', 'Scrub Radius [in]', 
          'Mechanical Trail [in]', 'Tie-rod Moment Arm [in]', 'Coefficient of Friction']
base_vals = [caster_angle_deg, kpi_angle_deg, scrub_radius_in, 
             mech_trail_in, tierod_moment_arm_in, mu]
delta_pct  = 0.25    # ±25% variation
ref_idx = int(np.round((n_steps - 1) * 105 / max_sw_angle_deg))

colors_sens = plt.cm.tab10(np.linspace(0, 1, 10))

for sp, ax_s in enumerate(axs.flatten()):
    style_ax(ax_s)
    
    sweep_vals = np.linspace(base_vals[sp]*(1-delta_pct), base_vals[sp]*(1+delta_pct), 20)
    efforts    = np.zeros_like(sweep_vals)
    
    for sv in range(len(sweep_vals)):
        p = list(base_vals)
        p[sp] = sweep_vals[sv]
        
        cas_r   = p[0] * deg2rad
        kpi_r   = p[1] * deg2rad
        scr     = p[2]
        mtr     = p[3]
        tarm    = p[4]
        mu_s    = p[5]
        
        wa_r = np.arcsin(min(rack_disp_in[ref_idx] / tarm, 0.9999))
        
        T_a = mu_s * W_per_front_lbf * (mtr + pneum_trail_in) * np.cos(wa_r) + \
              mu_s * W_per_front_lbf * scr * np.sin(wa_r)
        T_g = W_per_front_lbf * mtr * np.sin(wa_r) * np.sin(cas_r) + \
              W_per_front_lbf * scr * np.sin(wa_r) * np.sin(kpi_r) * np.cos(cas_r)
        T_kp = T_a + T_g
        tarm_eff = tarm * np.cos(wa_r)
        F_r  = (T_kp / tarm_eff) / np.cos(toe_link_rad)
        T_tot = (F_r * pinion_radius_in + col_friction_inlb) * inlb2Nm
        efforts[sv] = T_tot
        
    xn = sweep_vals / base_vals[sp]   # normalised x axis
    ax_s.plot(xn, efforts, '-o', color=colors_sens[sp], linewidth=2, markersize=4, markerfacecolor=colors_sens[sp])
    
    ax_s.axvline(1, color=clr_text, linestyle='--', alpha=0.5, linewidth=1)
    ax_s.axhline(6, color=(0.50, 0.95, 0.65), linestyle='--', linewidth=0.8, alpha=0.5)
    ax_s.axhline(15, color=(0.50, 0.95, 0.65), linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax_s.set_xlabel('Parameter Multiple of Baseline', fontsize=9)
    ax_s.set_ylabel('Effort @ 105° SW [N·m]', fontsize=9)
    ax_s.set_title(params[sp], fontsize=10, fontweight='bold')

fig2.suptitle('Sensitivity Analysis – Effort at 105° Steering Wheel Angle', fontsize=13, fontweight='bold', color=clr_text)
plt.tight_layout()

# =========================================================================
#  SECTION 14 – EXPORT TABLE TO CSV
# =========================================================================

export_data = {
    'SW_Angle_deg': sw_angles_deg,
    'Wheel_Angle_deg': wheel_angle_deg,
    'T_Trail_Nm': T_trail_Nm,
    'T_Scrub_Nm': T_scrub_Nm,
    'T_Caster_Nm': T_caster_Nm,
    'T_KPI_Nm': T_kpi_Nm,
    'T_Align_Nm': T_align_Nm,
    'T_Geom_Nm': T_geom_Nm,
    'T_Pinion_Nm': T_pinion_Nm,
    'T_ColFriction_Nm': T_col_Nm * np.ones(n_steps),
    'T_Total_Nm': T_total_Nm,
    'F_Rack_N': F_rack_N
}

df_export = pd.DataFrame(export_data)
df_export.to_csv('steering_effort_results.csv', index=False)
print('\n✓ Results exported to steering_effort_results.csv')

# =========================================================================
#  SECTION 15 – FINAL VALIDATION CHECK
# =========================================================================

i0   = np.argmin(np.abs(sw_angles_deg - 0))
i105 = np.argmin(np.abs(sw_angles_deg - 105))

print('\n=== TARGET VALIDATION ===')
print(f'  Effort at   0° SW : {T_total_Nm[i0]:.3f} N·m  ', end='')
if 6 <= T_total_Nm[i0] <= 15:
    print('[✓ Within 6–15 N·m target]')
else:
    print('[! Outside 6–15 N·m target]')

print(f'  Effort at 105° SW : {T_total_Nm[i105]:.3f} N·m  ', end='')
if 6 <= T_total_Nm[i105] <= 15:
    print('[✓ Within 6–15 N·m target]')
else:
    print('[! Outside 6–15 N·m target]')

print('\n=== RUN COMPLETE ===')

# Display plots
plt.show()

# END OF FILE