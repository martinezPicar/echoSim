import numpy as np
import matplotlib.pyplot as plt

def plot_g3plx_curves(mode='epsilon', duration=5.0, f_center=1500.0):
    """
    Direct geometrical simulation of G3PLX's specular reflection model.
    Traces the exact Doppler frequency vs. Time curves.
    """
    t = np.linspace(0, duration, 1000)
    z = np.linspace(-1.0, 1.0, 2000)  # Trail length
    dz = z[1] - z[0]

    # 1. Define physical trail shape deformed by wind shear
    if mode == 'C_shape':
        x = np.sin(np.pi * z)
    else:  # 'epsilon'
        # Fundamental + 3rd order harmonic (G3PLX 3rd-order curve)
        x = np.sin(np.pi * z) + 0.35 * np.sin(3.0 * np.pi * z)

    # Tangent / Slope along the trail dx/dz
    dxdz = np.gradient(x, dz)

    # Trail rotates/drifts across aspect angle over time
    t_center = duration / 2.0
    rotation_angle = 1.0 * (t - t_center)  # Rotation over time

    specular_times = []
    specular_freqs = []

    # 2. Find points along the trail where slope equals aspect angle (Specular condition)
    for ti, angle in zip(t, rotation_angle):
        tan_angle = np.tan(angle)
        
        # Specular points occur where slope matches tan(angle)
        diff = np.abs(dxdz - tan_angle)
        
        # Threshold to extract specular reflecting regions
        specular_indices = np.where(diff < 0.05)[0]

        for idx in specular_indices:
            # Doppler shift is proportional to local velocity x(z)
            doppler_hz = 20.0 * x[idx] * np.cos(angle)
            specular_times.append(ti)
            specular_freqs.append(f_center + doppler_hz)

    # 3. Plot in SpectrumLab style
    plt.figure(figsize=(10, 5), facecolor='black')
    ax = plt.axes()
    ax.set_facecolor('black')

    plt.scatter(
        specular_times, 
        specular_freqs, 
        s=2.5, 
        c='yellow', 
        edgecolor='none', 
        alpha=0.8
    )

    plt.title(f"G3PLX Meteor Simulation — Trace Mode: '{mode}'", color='white', fontsize=12)
    plt.xlabel("Time (seconds)", color='white')
    plt.ylabel("Frequency (Hz)", color='white')
    plt.ylim(f_center - 30, f_center + 30)
    plt.grid(True, color='gray', alpha=0.3)
    
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.tight_layout()
    plt.show()

# Run both traces
plot_g3plx_curves(mode='C_shape')
plot_g3plx_curves(mode='epsilon')