import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# --- Configuration ---
FREQ_BINS = 512          # Number of frequency bins in the spectrogram
TIME_BINS = 200          # Number of time bins (pixels) to keep in history
PING_MEAN_INTERVAL = 30  # Average time (frames) between simulated pings
PING_MAX_INTENSITY = 2.0 # Maximum intensity of a ping (in arbitrary units)
NOISE_FLOOR = 0.2        # Background noise level

# --- Data Initialization ---
# This will hold the history of the spectrogram. New data is added to the end,
# and old data "falls off" the beginning.
# We'll use a simple 2D numpy array.
waterfall_data = np.full((TIME_BINS, FREQ_BINS), NOISE_FLOOR)
time_since_last_ping = 0

def generate_spectrum_line():
    """Generates a new line of spectrum data, possibly containing a simulated ping."""
    global time_since_last_ping
    
    # Start with the noise floor and add some random noise
    line = np.random.normal(NOISE_FLOOR, 0.05, FREQ_BINS)
    line = np.clip(line, 0, None) # Ensure values are non-negative
    
    # Decide if a ping will occur in this time step
    time_since_last_ping += 1
    if time_since_last_ping >= random.expovariate(1.0 / PING_MEAN_INTERVAL):
        # A new ping! Reset the timer.
        time_since_last_ping = 0
        
        # Simulate a ping: a sharp spike in the spectrum at a random frequency
        ping_frequency = random.randint(10, FREQ_BINS - 10)
        ping_width = random.randint(1, 5)
        ping_intensity = random.uniform(0.5, PING_MAX_INTENSITY)
        
        # Add the ping to the spectrum line
        start = max(0, ping_frequency - ping_width)
        end = min(FREQ_BINS, ping_frequency + ping_width)
        line[start:end] += ping_intensity
        
    return line

def update_plot(frame):
    """Updates the waterfall plot with a new line of data."""
    global waterfall_data
    
    # Generate a new spectrum line
    new_line = generate_spectrum_line()
    
    # Add the new line to the waterfall data and remove the oldest line
    waterfall_data = np.roll(waterfall_data, shift=-1, axis=0)
    waterfall_data[-1, :] = new_line
    
    # Update the image data
    img.set_array(waterfall_data)
    
    return img,

# --- Set up the plot ---
fig, ax = plt.subplots()
# Display the waterfall data. Frequency is on the x-axis, time on the y-axis.
# The most recent data is at the bottom.
img = ax.imshow(waterfall_data, aspect='auto', origin='lower', 
                extent=[0, FREQ_BINS, 0, TIME_BINS])
ax.set_xlabel("Frequency (arbitrary units)")
ax.set_ylabel("Time (frames ago)")
ax.set_title("Meteor Ping Simulator")

# --- Run the animation ---
# The 'interval' parameter controls the speed of the simulation (in milliseconds)
ani = FuncAnimation(fig, update_plot, interval=50, blit=True)

plt.show()