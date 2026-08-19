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


def simulate_delayed_diffusion_short_epsilon_no_beacon(
    sample_rate=44100,
    duration=15.0,        # Simulation duration
    carrier_freq=1000.0,  # Center reference frequency for Doppler display
    snr_db=18.0
):
    """
    Simulates a 3rd-order wind-sheared overdense meteor echo (NO DIRECT BEACON):
    - Phase 1 (t = 2.0s to 2.1s): 0.1s initial trunk with ±4.5 Hz Doppler spread.
    - Full Reflection (t = 2.0s to 5.0s): Echo stays at full strength for 3 seconds.
    - Delayed Diffusion (t > 5.0s): Ambipolar diffusion decay starts after 3s, fading signal.
    - Reduced 3rd-Order Shear: Coefficient of 0.15 for clean Epsilon branching.
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

    # --- 3. Spatial Trail & Reduced 3rd-Order Shear Field ---
    nz = 4000
    z = np.linspace(-1.2, 1.2, nz)
    dz = z[1] - z[0]

    # Initial straight cylinder slope
    dx0_dz = np.gradient(0.01 * z, dz)

    # Reduced 3rd-Order Sinusoidal Profile
    v_wind = 80.0 * (np.sin(np.pi * z) + 0.15 * np.sin(3.0 * np.pi * z))
    dv_dz = np.gradient(v_wind, dz)

    # --- 4. Signal Synthesis Loop ---
    phase_z = np.zeros(nz)
    echo_signal = np.zeros(num_samples)

    trunk_duration = 0.10  # 100 ms initial entry phase

    for i in range(num_samples):
        if not active_mask[i]:
            continue

        tau = t_active[i]

        if tau < trunk_duration:
            # PHASE 1: Straight-trail entry phase (0.1s Initial Trunk)
            doppler_z = 4.5 * np.sin(2.0 * np.pi * z)
            specular_weight = np.exp(-(dx0_dz ** 2) / 0.008)

        else:
            # PHASE 2: Wind Shear Deformation
            shear_tau = tau - trunk_duration
            shear_factor = 0.22 * (1.0 - np.exp(-shear_tau / 1.5))
            dxdz_t = dx0_dz + (dv_dz * shear_factor)

            # Thermal diffusion broadening expands specular aperture only after 3 seconds
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
    DURATION = 15.0       # 15-second simulation window
    CARRIER_FREQ = 1000.0

    print("Generating Echo (No Direct Beacon Signal)...")
    t, audio = simulate_delayed_diffusion_short_epsilon_no_beacon(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        carrier_freq=CARRIER_FREQ,
        snr_db=18.0
    )

    # 1. Save WAV Audio
    wav_file = "brams_no_beacon_epsilon.wav"
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

    plt.title("BRAMS Network — Meteor Echo (No Direct Beacon Carrier)", color='white', fontsize=11)
    plt.xlabel("Time (seconds)", color='white')
    plt.ylabel("Doppler Offset (Hz) [Relative to Reference Frequency]", color='white')
    plt.ylim(CARRIER_FREQ - 100, CARRIER_FREQ + 100)
    plt.xlim(0, 12)  # Focus on event window (t = 0s to 12s)

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
    img_filename = "brams_no_beacon_spectrogram.png"
    plt.savefig(img_filename, dpi=300, facecolor='black', edgecolor='none')
    print(f"Saved spectrogram plot to '{img_filename}'")

    plt.show()


if __name__ == "__main__":
    main()