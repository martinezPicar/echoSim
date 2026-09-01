import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

# Set up figure
fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)

# Geometry parameters
d = 5.0
h_spec = 4.0

tx = np.array([-d, 0.0])
rx = np.array([d, 0.0])
spec = np.array([0.0, h_spec])

# Plot ground line
ax.plot([-d - 1.5, d + 1.5], [0, 0], 'k-', lw=1.5)

# Plot key locations
ax.plot(tx[0], tx[1], 'ko', markersize=7)
ax.plot(rx[0], rx[1], 'ko', markersize=7)
ax.plot(spec[0], spec[1], 'ko', markersize=7)

# Point labels
ax.text(tx[0], tx[1] - 0.35, 'Tx', fontsize=12, ha='center', va='top', fontweight='bold')
ax.text(rx[0], rx[1] - 0.35, 'Rx', fontsize=12, ha='center', va='top', fontweight='bold')
ax.text(spec[0] - 0.25, spec[1], r'$h_{spec}$', fontsize=12, ha='right', va='center')

# Vectors r_t (red) and r_r (violet)
ax.plot([tx[0], spec[0]], [tx[1], spec[1]], color='red', lw=2.5)
ax.plot([spec[0], rx[0]], [spec[1], rx[1]], color='violet', lw=2.5)

# Black text labels for vectors
mid_t = (tx + spec) / 2
mid_r = (rx + spec) / 2
ax.text(mid_t[0] - 0.3, mid_t[1] + 0.2, r'$r_t$', fontsize=13, color='black', ha='right', va='bottom')
ax.text(mid_r[0] + 0.3, mid_r[1] + 0.2, r'$r_r$', fontsize=13, color='black', ha='left', va='bottom')

# Bistatic bisector line
bisector_top = spec[1] + 2.2
bisector_bottom = spec[1] - 2.0
ax.plot([0, 0], [bisector_bottom, bisector_top], color='gray', linestyle=':', lw=1.5)
ax.text(-0.2, bisector_top + 0.1, r'Bistatic Bisector ($\beta$)', fontsize=11, color='black', ha='right', va='bottom')

# Meteor Trail (discontinuous line forming 12° angle with vertical)
theta_deg = 12
theta_rad = np.radians(theta_deg)
trail_len = 2.8

dx = trail_len * np.sin(theta_rad)
dy = trail_len * np.cos(theta_rad)

trail_x = [spec[0] - dx, spec[0] + dx]
trail_y = [spec[1] - dy, spec[1] + dy]

ax.plot(trail_x, trail_y, color='black', linestyle='--', lw=2.0)
ax.text(spec[0] + dx + 0.15, spec[1] + dy + 0.1, 'Trail Skew', fontsize=11, color='black', ha='left', va='bottom')

# Angle arc indicating trail skew relative to vertical bisector
arc = Arc((0, h_spec), width=1.8, height=1.8, angle=0, theta1=90 - theta_deg, theta2=90, color='gray', lw=1.2)
ax.add_patch(arc)

# Axis labels and formatting
ax.set_ylabel('Altitude', fontsize=13, labelpad=10)
ax.set_xlabel('Horizontal distance', fontsize=13, labelpad=10)

# Remove numerical ticks/values
ax.set_xticks([])
ax.set_yticks([])

ax.set_aspect('equal')
ax.set_xlim(-d - 1.5, d + 1.5)
ax.set_ylim(-0.8, h_spec + 3.0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

plt.title('Simplified Bistatic Forward-Scatter Geometry', fontsize=14, pad=18, fontweight='bold')
plt.tight_layout()
plt.show()