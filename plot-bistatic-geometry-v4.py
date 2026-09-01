import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

# Set up figure
fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)

# Geometry parameters
d = 5.0  # Horizontal distance d_x
h = 4.0  # Height h_spec

tx = np.array([-d, 0.0])
rx = np.array([d, 0.0])
spec = np.array([0.0, h])

# Earth surface line
ground_x = [-d - 1.5, d + 1.5]
ax.plot(ground_x, [0, 0], 'k-', lw=1.5)
#ax.text(-d / 2, 0.25, 'Earth\nsurface', fontsize=10, ha='center', va='bottom')
ax.text(-d / 2, 0.1, 'Earth\nsurface', fontsize=11, ha='center', va='bottom')

# Points: Tx, Rx, Specular Point (NO circle at ground projection)
ax.plot(tx[0], tx[1], 'ko', markersize=7, zorder=5)
ax.plot(rx[0], rx[1], 'ko', markersize=7, zorder=5)
ax.plot(spec[0], spec[1], 'ko', markersize=7, zorder=5)

# Labels for Tx, Rx, and Specular Point Horizontal Projection
ax.text(tx[0], tx[1] - 0.35, 'Tx', fontsize=11, ha='center', va='top', fontweight='bold')
ax.text(rx[0], rx[1] - 0.35, 'Rx', fontsize=11, ha='center', va='top', fontweight='bold')
ax.text(0.0, -0.35, "Horizontal distance", fontsize=11, ha='center', va='top')

# Specular point label
#ax.text(spec[0] - 0.25, spec[1] + 0.1, "Specular\npoint", fontsize=10, ha='right', va='center')
ax.text(spec[0] - 0.3, spec[1] + 0.25, "Specular\npoint", fontsize=11, ha='right', va='center')

# Vectors r_t (red) and r_r (violet)
ax.plot([tx[0], spec[0]], [tx[1], spec[1]], color='red', lw=2.5)
ax.plot([spec[0], rx[0]], [spec[1], rx[1]], color='violet', lw=2.5)

# Vector labels r_t and r_r
mid_t = (tx + spec) / 2
mid_r = (rx + spec) / 2
ax.text(mid_t[0] - 0.3, mid_t[1] + 0.2, r'$\mathbf{r}_\mathrm{t}$', fontsize=13, color='black', ha='right', va='bottom')
ax.text(mid_r[0] + 0.3, mid_r[1] + 0.2, r'$\mathbf{r}_\mathrm{r}$', fontsize=13, color='black', ha='left', va='bottom')

# Vertical dotted line (Bisector reference axis from ground up)
bisector_top = spec[1] + 2.5
ax.plot([0, 0], [0, bisector_top], color='black', linestyle=':', lw=1.5)

# Height dimension arrow h_spec
#ax.annotate('', xy=(0.5, 0), xytext=(0.5, h),
#            arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
ax.annotate('', xy=(5.35, 0), xytext=(5.35, h),arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
ax.text(5.5, h / 2, r'$h_{spec}$', fontsize=11, ha='left', va='center')

# Meteor trail (dashed line at 12° skew angle relative to vertical)
theta_deg = 12
theta_rad = np.radians(theta_deg)
trail_len = 2.8

dx_trail = trail_len * np.sin(theta_rad)
dy_trail = trail_len * np.cos(theta_rad)

trail_x = [spec[0] - dx_trail, spec[0] + dx_trail]
trail_y = [spec[1] - dy_trail, spec[1] + dy_trail]

ax.plot(trail_x, trail_y, color='black', linestyle='--', lw=2.0)
#ax.text(spec[0] + dx_trail + 0.15, spec[1] + dy_trail + 0.1, 'Meteor Trail', fontsize=10, color='black', ha='left', va='bottom')
ax.text(spec[0] + dx_trail + 0.15, spec[1] + dy_trail, 'Meteor Trail', fontsize=11, color='black', ha='left', va='bottom')

# Arc for beta angle (between r_t vector and vertical bisector)
beta_angle = np.degrees(np.arctan2(d, h))
arc_beta = Arc((0, h), width=2.2, height=2.2, angle=0, 
               theta1=270 - beta_angle, theta2=270, color='gray', lw=1.5)
ax.add_patch(arc_beta)
# ax.text(-0.6, h - 1.2, r'$\beta$', fontsize=10, color='black', ha='center', va='center')
ax.text(-0.7, h - 1.25, r'$\beta$', fontsize=11, color='black', ha='center', va='center')

# Arc for Skew Angle (strictly from vertical dotted line [90 deg] to meteor trail [90 - 12 deg])
arc_skew = Arc((0, h), width=3.55, height=3.5, angle=0,theta1=90 - theta_deg, theta2=90, color='gray', lw=1.5)
#arc_skew = Arc((0, h), width=2.5, height=2.5, angle=0, 
#               theta1=90 - theta_deg, theta2=90, color='gray', lw=1.5)
ax.add_patch(arc_skew)

# Place text "Skew Angle" clearly next to the small arc
#ax.text(0.3, h + 1.4, 'Skew Angle', fontsize=10, color='black', ha='left', va='bottom')
ax.text(0.5, h + 1.5, 'Skew\nAngle $\sigma$', fontsize=10, color='black', ha='left', va='bottom')

# Dimension lines for horizontal symmetry (d_x)
#y_dim = -1.2
#ax.annotate('', xy=(-d, y_dim), xytext=(0, y_dim),
#            arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
#ax.annotate('', xy=(0, y_dim), xytext=(d, y_dim),
#            arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))

# Vertical ticks at ends of dimension lines
# ax.plot([-d, -d], [y_dim - 0.2, y_dim + 0.2], 'k-', lw=1.2)
# ax.plot([0, 0], [y_dim - 0.2, y_dim + 0.2], 'k-', lw=1.2)
# ax.plot([d, d], [y_dim - 0.2, y_dim + 0.2], 'k-', lw=1.2)

# ax.text(-d / 2, y_dim, r'$d_x$', fontsize=12, ha='center', va='center', backgroundcolor='white')
# ax.text(d / 2, y_dim, r'$d_x$', fontsize=12, ha='center', va='center', backgroundcolor='white')

# Axis limits and framing setup
ax.set_aspect('equal')
ax.set_xlim(-d - 1.5, d + 1.5)
ax.set_ylim(-2.0, h + 3.2)
ax.axis('off')

plt.title('Simplified Bistatic Forward-Scatter Geometry', fontsize=14, pad=15, fontweight='bold')
plt.tight_layout()
plt.savefig('corrected_skew_geometry.png', dpi=300, bbox_inches='tight')
plt.show()