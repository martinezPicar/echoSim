import numpy as np
import matplotlib.pyplot as plt

# Define geometry parameters
h_spec = 90  # Specular point height in km
tx_x = -200  # Transmitter horizontal position (km)
rx_x = 200   # Receiver horizontal position (km)
tx_h = 0     # Transmitter height (km, ground level)
rx_h = 0     # Receiver height (km, ground level)

# Meteor trail angle from vertical (degrees)
meteor_angle_deg = 12
meteor_angle_rad = np.radians(meteor_angle_deg)

# Specular point coordinates
spec_x = 0
spec_h = h_spec

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Plot transmitter
ax.plot(tx_x, tx_h, 'bs', markersize=12, label='Transmitter (Tx)')
ax.text(tx_x-15, tx_h-5, 'Tx', fontsize=12, fontweight='bold')

# Plot receiver
ax.plot(rx_x, rx_h, 'go', markersize=12, label='Receiver (Rx)')
ax.text(rx_x+5, rx_h-5, 'Rx', fontsize=12, fontweight='bold')

# Plot specular point
ax.plot(spec_x, spec_h, 'k*', markersize=20, label='Specular Point')
ax.text(spec_x+5, spec_h+5, f'h_spec = {h_spec} km', fontsize=11)

# Draw vectors to specular point
# Tx to specular point (r_t in red)
ax.plot([tx_x, spec_x], [tx_h, spec_h], 'r-', linewidth=2.5, label='$r_t$ (Tx to specular)')
# Specular point to Rx (r_r in violet)
ax.plot([spec_x, rx_x], [spec_h, rx_h], 'violet', linewidth=2.5, label='$r_r$ (specular to Rx)')

# Calculate and draw bistatic bisector (beta)
# Bisector direction vector from specular point
# Normalize vectors from specular point to Tx and Rx
v_tx = np.array([tx_x - spec_x, tx_h - spec_h])
v_rx = np.array([rx_x - spec_x, rx_h - spec_h])
v_tx_norm = v_tx / np.linalg.norm(v_tx)
v_rx_norm = v_rx / np.linalg.norm(v_rx)

# Bisector direction (sum of normalized vectors)
bisector = v_tx_norm + v_rx_norm
bisector = bisector / np.linalg.norm(bisector) * 80  # Scale for visualization

# Plot bisector
ax.arrow(spec_x, spec_h, bisector[0], bisector[1], 
         head_width=5, head_length=5, fc='orange', ec='orange', 
         linewidth=2, label='Bistatic Bisector ($\\beta$)')

# Add angle annotation for beta
# Calculate angle between vectors
angle_tx = np.arctan2(v_tx[1], v_tx[0])
angle_rx = np.arctan2(v_rx[1], v_rx[0])

# Add arc for beta angle
theta = np.linspace(angle_tx, angle_rx, 30)
r_arc = 25
arc_x = spec_x + r_arc * np.cos(theta)
arc_y = spec_h + r_arc * np.sin(theta)
ax.plot(arc_x, arc_y, 'orange', linewidth=2, linestyle='--')

# Label beta angle
mid_angle = (angle_tx + angle_rx) / 2
label_x = spec_x + (r_arc + 10) * np.cos(mid_angle)
label_y = spec_h + (r_arc + 10) * np.sin(mid_angle)
ax.text(label_x, label_y, '$\\beta$', fontsize=14, color='orange', fontweight='bold')

# Add meteor trail (dashed line at 12° from vertical)
# Direction: meteor trail is tilted from vertical
# Choose direction: tilted to the right (can be changed)
meteor_length = 150  # Length of meteor trail in km
# Direction vector for meteor (12° from vertical, tilted right)
# Vertical direction is (0, 1), rotate by 12° clockwise
meteor_dir = np.array([np.sin(meteor_angle_rad), np.cos(meteor_angle_rad)])
# Extend in both directions from specular point
meteor_start = np.array([spec_x, spec_h]) - meteor_dir * (meteor_length/2)
meteor_end = np.array([spec_x, spec_h]) + meteor_dir * (meteor_length/2)

# Plot meteor trail
ax.plot([meteor_start[0], meteor_end[0]], 
        [meteor_start[1], meteor_end[1]], 
        '--', color='gray', linewidth=3, alpha=0.8, label='Meteor trail')

# Add meteor trail annotation with angle
mid_meteor_x = (meteor_start[0] + meteor_end[0]) / 2
mid_meteor_y = (meteor_start[1] + meteor_end[1]) / 2

# Add small angle annotation for meteor tilt
# Draw a small vertical line at specular point for reference
vert_line_length = 30
vert_start = np.array([spec_x, spec_h - vert_line_length/2])
vert_end = np.array([spec_x, spec_h + vert_line_length/2])
ax.plot([vert_start[0], vert_end[0]], 
        [vert_start[1], vert_end[1]], 
        ':', color='gray', linewidth=1, alpha=0.5)

# Draw arc for meteor angle
angle_arc_radius = 20
angle_arc_theta = np.linspace(np.pi/2 - meteor_angle_rad, np.pi/2, 20)
arc_x_meteor = spec_x + angle_arc_radius * np.cos(angle_arc_theta)
arc_y_meteor = spec_h + angle_arc_radius * np.sin(angle_arc_theta)
ax.plot(arc_x_meteor, arc_y_meteor, 'gray', linewidth=1.5, alpha=0.6)

# Label the angle
label_angle_pos = np.pi/2 - meteor_angle_rad/2
label_x_angle = spec_x + (angle_arc_radius + 8) * np.cos(label_angle_pos)
label_y_angle = spec_h + (angle_arc_radius + 8) * np.sin(label_angle_pos)
ax.text(label_x_angle, label_y_angle, f'12°', fontsize=10, color='gray', alpha=0.8)

# Add text annotation along meteor trail
ax.text(meteor_end[0]+5, meteor_end[1]-5, 'Meteor trail', 
        fontsize=10, color='gray', alpha=0.8, rotation=12)

# Add labels for vectors
# Label for r_t
mid_x_t = (tx_x + spec_x) / 2
mid_h_t = (tx_h + spec_h) / 2
ax.text(mid_x_t-30, mid_h_t+5, '$r_t$', fontsize=14, color='red', fontweight='bold')

# Label for r_r
mid_x_r = (spec_x + rx_x) / 2
mid_h_r = (spec_h + rx_h) / 2
ax.text(mid_x_r+5, mid_h_r+5, '$r_r$', fontsize=14, color='violet', fontweight='bold')

# Add specular point indicator (horizontal line at specular height)
ax.axhline(y=spec_h, xmin=0, xmax=spec_x/400 + 0.5, 
           linestyle=':', color='k', alpha=0.3)

# Set labels and title
ax.set_xlabel('Horizontal Distance (km)', fontsize=12)
ax.set_ylabel('Altitude (km)', fontsize=12)
ax.set_title('Bistatic Forward-Scatter Geometry with Tilted Meteor Trail', 
             fontsize=14, fontweight='bold')

# Set axis limits with some padding
x_pad = 50
y_pad = 20
ax.set_xlim(tx_x - x_pad, rx_x + x_pad)
ax.set_ylim(-y_pad, h_spec + y_pad*2)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Add legend
ax.legend(loc='upper left', fontsize=11)

# Add earth surface indicator
ax.axhline(y=0, color='brown', linewidth=2, alpha=0.5)
ax.text(rx_x-30, -5, 'Earth\'s Surface', fontsize=10, color='brown')

# Show plot
plt.tight_layout()
plt.show()