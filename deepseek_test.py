#!/usr/bin/env python3
"""
Ultra-simple test - just draws a moving line
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Interactive mode
plt.ion()

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Create data
data = np.zeros((100, 200))
im = ax.imshow(data, aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=1)
ax.set_title("Simple Moving Pattern - You should see a diagonal line")
plt.colorbar(im)

fig.show()
fig.canvas.draw()

print("Starting animation... Press Ctrl+C to stop")

try:
    pos = 0
    while plt.fignum_exists(fig.number):
        # Create a diagonal line
        data = np.zeros((100, 200))
        for i in range(100):
            # Diagonal line that moves
            idx = int(i * 200 / 100 + pos) % 200
            if 0 <= idx < 200:
                data[i, idx] = 1.0
            # Add some random noise
            data[i, :] += np.random.normal(0, 0.02, 200)
        
        # Normalize
        data = data / data.max()
        
        # Update display
        im.set_array(data)
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        pos += 1
        time.sleep(0.05)
        
except KeyboardInterrupt:
    print("\nStopping...")

plt.close(fig)
print("Done!")