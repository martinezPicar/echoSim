"""
Simplified bistatic forward-scatter geometry (schematic, no numeric values shown).

Shows:
  - Tx and Rx on the ground baseline, equidistant from the specular point
  - Specular point S centered horizontally (x = 0) at altitude h_spec
  - r_t : vector Tx -> S   (red line, black label)
  - r_r : vector S  -> Rx  (violet line, black label)
  - beta: bistatic angle (angle between r_t and r_r at S) with its bisector
  - meteor trail: dashed line through S, tilted 12 deg from vertical
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------------------------------------------------------------------
# Geometry parameters (schematic only — no values are displayed on the plot)
# ----------------------------------------------------------------------
h_spec = 100.0          # altitude of the specular point
half_baseline = 300.0   # Tx and Rx are equidistant from x = 0
xt, yt = -half_baseline, 0.0   # Tx position on the ground baseline
xr, yr = half_baseline, 0.0    # Rx position on the ground baseline
xs, ys = 0.0, h_spec            # specular point, centered at x = 0

meteor_tilt_deg = 12.0  # meteor trail tilt from the local vertical

# ----------------------------------------------------------------------
# Derived vectors
# ----------------------------------------------------------------------
a = np.array([xt - xs, yt - ys])       # S -> Tx
b = np.array([xr - xs, yr - ys])       # S -> Rx
a_hat, b_hat = a / np.linalg.norm(a), b / np.linalg.norm(b)

bis_dir = (a_hat + b_hat)
bis_dir = bis_dir / np.linalg.norm(bis_dir)   # bisector direction (points down, between Tx & Rx)

# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.5))

# Ground
x_lo, x_hi = xt - 60, xr + 60
ax.axhline(0, color='#6b4a2b', lw=1.5, zorder=1)
ax.fill_between([x_lo, x_hi], -12, 0, color='#d9c9a3', alpha=0.6, zorder=0)

# --- r_t (Tx -> S), red line, black label --------------------------------
ax.annotate('', xy=(xs, ys), xytext=(xt, yt),
            arrowprops=dict(arrowstyle='-|>', color='red', lw=2.2,
                             shrinkA=0, shrinkB=0), zorder=4)
mid_t = (np.array([xt, yt]) + np.array([xs, ys])) / 2
ax.text(mid_t[0] - 18, mid_t[1] + 8, r'$r_t$', color='black', fontsize=15,
         fontweight='bold', ha='center')

# --- r_r (S -> Rx), violet line, black label ------------------------------
ax.annotate('', xy=(xr, yr), xytext=(xs, ys),
            arrowprops=dict(arrowstyle='-|>', color='darkviolet', lw=2.2,
                             shrinkA=0, shrinkB=0), zorder=4)
mid_r = (np.array([xs, ys]) + np.array([xr, yr])) / 2
ax.text(mid_r[0] + 18, mid_r[1] + 8, r'$r_r$', color='black', fontsize=15,
         fontweight='bold', ha='center')

# --- Tx / Rx / S markers --------------------------------------------------
ax.plot(xt, yt, marker='^', color='black', ms=12, zorder=5)
ax.text(xt, yt - 10, 'Tx', ha='center', va='top', fontsize=12, fontweight='bold')

ax.plot(xr, yr, marker='^', color='black', ms=12, zorder=5)
ax.text(xr, yr - 10, 'Rx', ha='center', va='top', fontsize=12, fontweight='bold')

ax.plot(xs, ys, marker='o', color='black', ms=7, zorder=6)
ax.text(xs + 10, ys + 6, r'specular point ($h_{spec}$)', fontsize=11, ha='left', color='black')

# --- vertical (local zenith) reference line at S --------------------------
v_len = 55
ax.plot([xs, xs], [ys, ys + v_len], ls=':', color='gray', lw=1.2, zorder=2)
ax.plot([xs, xs], [0, ys], ls=':', color='gray', lw=1.0, zorder=1)

# --- h_spec marker (no numeric value, just the symbol) ---------------------
ax.annotate('', xy=(xt - 25, ys), xytext=(xt - 25, 0),
            arrowprops=dict(arrowstyle='<->', color='dimgray', lw=1.2))
ax.text(xt - 32, ys / 2, r'$h_{spec}$', rotation=90, va='center', ha='right',
         fontsize=12, color='dimgray')

# --- bistatic angle beta: arc between S->Tx and S->Rx directions ----------
ang_a = np.degrees(np.arctan2(a_hat[1], a_hat[0]))
ang_b = np.degrees(np.arctan2(b_hat[1], b_hat[0]))
arc_r = 45
arc = patches.Arc((xs, ys), 2 * arc_r, 2 * arc_r,
                   theta1=min(ang_a, ang_b), theta2=max(ang_a, ang_b),
                   color='black', lw=1.4, zorder=3)
ax.add_patch(arc)
beta_label_dir = bis_dir * (arc_r - 15)
ax.text(xs + beta_label_dir[0] - 12, ys + beta_label_dir[1], r'$\beta$',
         fontsize=16, ha='center', va='center', fontweight='bold', color='black')

# --- bisector of beta, dashed, extended a bit past S -----------------------
bis_len = 70
p_bis = np.array([xs, ys]) + bis_dir * bis_len
p_bis_back = np.array([xs, ys]) - bis_dir * 20
ax.plot([p_bis_back[0], p_bis[0]], [p_bis_back[1], p_bis[1]],
         ls='--', color='black', lw=1.3, zorder=3)
ax.text(p_bis[0] + 6, p_bis[1] - 18, 'bistatic bisector', fontsize=9,
         color='black', ha='left', style='italic')

# --- meteor trail: dashed, tilted 12 deg from vertical (no value shown) ----
tilt = np.radians(meteor_tilt_deg)
meteor_dir = np.array([np.sin(tilt), np.cos(tilt)])   # tilted from vertical
m_len = 60
p1 = np.array([xs, ys]) - meteor_dir * m_len
p2 = np.array([xs, ys]) + meteor_dir * m_len
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], ls=(0, (6, 4)), color='forestgreen',
         lw=2, zorder=5, label='meteor trail')
ax.text(p2[0] + 4, p2[1] + 2, 'meteor trail', fontsize=9,
         color='forestgreen', ha='left', va='bottom')

# small angle arc between meteor trail and local vertical (no degree label)
arc2_r = 22
arc2 = patches.Arc((xs, ys), 2 * arc2_r, 2 * arc2_r,
                    theta1=90 - meteor_tilt_deg, theta2=90,
                    color='forestgreen', lw=1.2, zorder=4)
ax.add_patch(arc2)

# ----------------------------------------------------------------------
# Cosmetics — no numeric values shown anywhere (axes ticks removed)
# ----------------------------------------------------------------------
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(-15, ys + 90)
ax.set_xlabel('Horizontal distance', fontsize=12)
ax.set_ylabel('Altitude', fontsize=12)
ax.set_title('Simplified bistatic forward-scatter geometry', fontsize=13, fontweight='bold')
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])
ax.axvline(0, color='0.85', lw=0.8, zorder=0)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig('bistatic_geometry.png', dpi=200)
