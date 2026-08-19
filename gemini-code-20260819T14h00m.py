import numpy as np
import scipy.io.wavfile as wav

# Ensure GUI backend compatibility on macOS/Conda environments
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


def simulate_meteor_with_dynamic_transition(
    sample_rate=44100,
    duration=15.0,        
    carrier_freq=1000.0,  # Audio carrier reference
    snr_db=18.0,
    trail_skew_angle_deg=12.0
):
    """
    Simulates an Overdense Meteor Echo with a 3-Phase Dynamic Evolution:
    - Phase 1 (t = 2.0s to 2.1s): 0.1s Initial Trunk (±4.5 Hz Doppler spread + initial skew).
    - Phase Transition (t = 2.1s to 2.3s): 0.2s Dynamic Morphing from trunk to wind profile.
    - Phase 2 (t > 2.3s): Fully developed fundamental + 3rd harmonic wind shear branches.
    - Delayed Diffusion (t > 5.0s): Ambipolar diffusion decay starts after 3 seconds of reflection.
    """
    num_samples = int(sample_rate * duration)
    dt = 1.0 / sample_rate
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # --- 1. Event Timing ---
    t_entry = 2.0  # Meteor impact at t = 2.0s
    active_mask = t >= t_entry
    t_active = np.maximum(0.0, t - t_entry)

    # --- 2. Delayed Ambipolar Diffusion Envelope ---
    rise_env = 1.0 - np.exp(-t_active / 0.01)
    
    # Delay diffusion decay so signal stays unattenuated for 3 seconds (t <= 5.0s)
    t_diffusion_start = 3.0  # Delay relative to t_entry
    diffusion_time = np.maximum(0.0, t_active - t_diffusion_start)
    decay_env = np.exp(-diffusion_time / 1.0)  # Smooth decay once t > 5.0s
    
    global_envelope = np.zeros(num_samples)
    global_envelope[active_mask] = rise_env[active_mask] * decay_env[active_mask]

    # --- 3. Spatial Trail Geometry & Multi-Harmonic Wind Field ---
    nz = 4000
    z = np.linspace(-1.2, 1.2, nz)
    dz = z[1] - z[0]

    # Initial spatial skew slope
    skew_rad = np.radians(trail_skew_angle_deg)
    initial_tilt_slope = np.tan(skew_rad)
    
    x0 = initial_tilt_slope * z
    dx0_dz = np.gradient(x0, dz)

    # Combined Wind Profile: Dominant Fundamental + Weak 3rd Harmonic
    v_fundamental = np.sin(np.pi * z)
    v_3rd_harmonic = 0.10 * np.sin(3.0 * np.pi * z)  # 10% weighting
    
    v_wind = 80.0 * (v_fundamental + v_3rd_harmonic)
    dv_dz = np.gradient(v_wind, dz)

    # --- 4. Signal Synthesis Loop ---
    phase_z = np.zeros(nz)
    echo_signal = np.zeros(num_samples)

    trunk_duration = 0.10       # 0.1s Initial Trunk (2.0s to 2.1s)
    transition_duration = 0.20  # 0.2s Dynamic Transition (2.1s to 2.3s)

    for i in range(num_samples):
        if not active_mask[i]:
            continue

        tau = t_active[i]

        if tau < trunk_duration:
            # PHASE 1: Skewed Entry Phase (0.1s Initial Trunk)
            doppler_z = 4.5 * np.sin(2.0 * np.pi * z) + (15.0 * initial_tilt_slope)
            specular_weight = 0.0 * np.exp(-((dx0_dz - initial_tilt_slope) ** 2) / 0.008) # 1.0

        elif tau < (trunk_duration + transition_duration):
            # DYNAMIC TRANSITION PHASE (0.2s: t = 2.1s to 2.3s)
            # Smooth S-curve transition factor alpha from 0.0 to 1.0
            alpha_linear = (tau - trunk_duration) / transition_duration
            alpha = 0.5 * (1.0 - np.cos(np.pi * alpha_linear))

            # Dynamic Doppler morphing
            doppler_trunk = 4.5 * np.sin(2.0 * np.pi * z) + (15.0 * initial_tilt_slope)
            doppler_z = (1.0 - alpha) * doppler_trunk + alpha * v_wind

            # Morphing trail derivative (bending starts forming)
            shear_factor = 0.22 * alpha
            dxdz_t = dx0_dz + (dv_dz * shear_factor)

            aperture = (1.0 - alpha) * 0.008 + alpha * 0.015
            specular_weight = np.exp(-(dxdz_t ** 2) / aperture)

        else:
            # PHASE 2: Fully Developed Wind Shear Bending
            shear_tau = tau - (trunk_duration + transition_duration)
            shear_factor = 0.22 * (1.0 - np.exp(-shear_tau / 1.5))
            
            dxdz_t = dx0_dz + (dv_dz * shear_factor)

            # Thermal diffusion broadening expands specular aperture after 3 seconds
            diffusion_tau = np.maximum(0.0, tau - t_diffusion_start)
            diffusion_aperture = 0.015 + (0.010 * diffusion_tau)
            specular_weight = np.exp(-(dxdz_t ** 2) / diffusion_aperture)

            doppler_z = v_wind

        # Instantaneous phase accumulation per altitude layer
        inst_freq_z = carrier_freq + doppler_z
        phase_z += 2.0 * np.pi * inst_freq_z * dt

        # Sum complex reflections across all specular regions
        reflections = specular_weight * np.sin(phase_z)
        echo_signal[i] = np.sum(reflections) * global_envelope[i]

    # Normalize echo signal strength
    if np.max(np.abs(echo_signal)) > 0:
        echo_signal = echo_signal / np.max(np.abs(echo_signal))

    # --- 5. Pure Meteor Echo + Background AWGN (No Beacon) ---
    snr_lin = 10 ** (snr_db / 10.0)
    noise = np.random.normal(0, np.sqrt(np.mean(echo_signal**2) / snr_lin), num_samples)

    final_audio = echo_signal + noise
    return t, final_audio / np.max(np.abs(final_audio))


def main():
    SAMPLE_RATE = 44100
    DURATION = 15.0       
    CARRIER_FREQ = 1000.0
    SKEW_ANGLE_DEG = 12.0

    print("Generating Echo with 0.2s Dynamic Transition Phase...")
    t, audio = simulate_meteor_with_dynamic_transition(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        carrier_freq=CARRIER_FREQ,
        snr_db=18.0,
        trail_skew_angle_deg=SKEW_ANGLE_DEG
    )

    # 1. Save WAV Audio
    wav_file = "brams_dynamic_transition_echo.wav"
    wav.write(wav_file, SAMPLE_RATE, np.int16(audio * 32767))
    print(f"Saved audio output to '{wav_file}'")

    # 2. Audio Playback (If sounddevice is present)
    if HAS_SOUNDDEVICE:
        sd.play(np.int16(audio * 32767), SAMPLE_RATE)

    # 3. Plot & Save Spectrogram Display
    plt.figure(figsize=(14, 6), facecolor='black')
    ax = plt.axes()
    ax.set_facecolor('navy')

    plt.specgram(
        audio,
        NFFT=8192,
        Fs=SAMPLE_RATE,
        noverlap=7168,
        cmap='jet',
        vmin=-60,
        vmax=-10
    )

    plt.title("BRAMS Network — Meteor Echo (0.1s Trunk → 0.2s Dynamic Transition → Harmonic Shear)", color='white', fontsize=11)
    plt.xlabel("Time (seconds)", color='white')
    plt.ylabel("Doppler Offset (Hz) [Relative to Reference Frequency]", color='white')
    plt.ylim(CARRIER_FREQ - 100, CARRIER_FREQ + 100)
    plt.xlim(0, 12)  

    # Custom Y-Axis Labels (-100 Hz to +100 Hz offset)
    ticks = np.linspace(CARRIER_FREQ - 100, CARRIER_FREQ + 100, 9)
    tick_labels = [f"{int(f - CARRIER_FREQ):+d} Hz" for f in ticks]
    plt.yticks(ticks, tick_labels, color='white')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.grid(True, color='cyan', alpha=0.18, linestyle='--')
    plt.tight_layout()

    # Save PNG directly to workspace
    img_filename = "brams_dynamic_transition_spectrogram.png"
    plt.savefig(img_filename, dpi=300, facecolor='black', edgecolor='none')
    print(f"Saved spectrogram plot to '{img_filename}'")

    plt.show()


if __name__ == "__main__":
    main()