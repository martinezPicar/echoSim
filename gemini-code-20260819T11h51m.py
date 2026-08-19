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


def simulate_3rd_order_short_echo(
    sample_rate=44100,
    duration=15.0,        # Reduced simulation length (reflection dies after 5s)
    carrier_freq=1000.0,  # Audio carrier representing BRAMS 49.970 MHz beacon
    snr_db=18.0
):
    """
    Simulates a 3rd-order wind-sheared overdense meteor echo:
    - Phase 1 (0 to 0.1s): 0.1s initial trunk (±4.5 Hz Doppler) at t = 2.0s to 2.1s.
    - Phase 2 (0.1s to 5.0s): 3rd-order sinusoidal shear deformation (Epsilon shape).
    - Diffusion Attenuation: Rapid ambipolar diffusion fades reflection completely by ~5s.
    """
    num_samples = int(sample_rate * duration)
    dt = 1.0 / sample_rate
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # --- 1. Event Timing ---
    t_entry = 2.0  # Meteor impact at t = 2.0s
    active_mask = t >= t_entry
    t_active = np.maximum(0.0, t - t_entry)

    # --- 2. Ambipolar Electron Diffusion Decay (5-Second Total Duration) ---
    # Fast formation (~0.01s) and aggressive exponential diffusion decay (tau = 1.2s)
    # The echo drops below noise floor within ~5 seconds after entry (t = 7.0s)
    rise_env = 1.0 - np.exp(-t_active / 0.01)
    diffusion_decay = np.exp(-t_active / 1.2)
    global_envelope = np.zeros(num_samples)
    global_envelope[active_mask] = rise_env[active_mask] * diffusion_decay[active_mask]

    # --- 3. Trail Discretization & 3rd-Order Wind Field ---
    nz = 4000
    z = np.linspace(-1.2, 1.2, nz)
    dz = z[1] - z[0]

    # Initial straight cylinder slope
    dx0_dz = np.gradient(0.01 * z, dz)

    # 3rd-Order Sinusoidal Shear Velocity Profile (Fundamental + 3rd Harmonic)
    v_wind = 80.0 * (np.sin(np.pi * z) + 0.45 * np.sin(3.0 * np.pi * z))
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
            # PHASE 2: Fast 3rd-Order Sinusoidal Wind Shear Deformation
            shear_tau = tau - trunk_duration
            shear_factor = 0.25 * (1.0 - np.exp(-shear_tau / 1.5))
            dxdz_t = dx0_dz + (dv_dz * shear_factor)

            # Thermal diffusion expands radius & lowers reflection efficiency over 5s
            diffusion_aperture = 0.015 + (0.008 * tau)
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

    # --- 5. Direct Reference Beacon Carrier ---
    beacon_signal = 0.035 * np.sin(2.0 * np.pi * carrier_freq * t)

    # Combine Echo + Direct Beacon + Noise
    total_sig = echo_signal + beacon_signal
    snr_lin = 10 ** (snr_db / 10.0)
    noise = np.random.normal(0, np.sqrt(np.mean(total_sig**2) / snr_lin), num_samples)

    final_audio = total_sig + noise
    return t, final_audio / np.max(np.abs(final_audio))


def main():
    SAMPLE_RATE = 44100
    DURATION = 15.0       # Focused 15-second simulation window
    CARRIER_FREQ = 1000.0

    print("Generating 3rd-Order Shear Echo (0.1s Trunk + 5s Diffusion Decay)...")
    t, audio = simulate_3rd_order_short_echo(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        carrier_freq=CARRIER_FREQ,
        snr_db=18.0
    )

    # 1. Save WAV Audio
    wav_file = "brams_3rd_order_5s_echo.wav"
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

    plt.title("BRAMS Network — 3rd-Order Shear Echo (0.1s Trunk, 5s Epsilon Lifetime)", color='white', fontsize=11)
    plt.xlabel("Time (seconds)", color='white')
    plt.ylabel("Doppler Offset (Hz) [Relative to Direct Beacon]", color='white')
    plt.ylim(CARRIER_FREQ - 100, CARRIER_FREQ + 100)
    plt.xlim(0, 12)  # Zoomed into event range (t = 0s to 12s)

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
    img_filename = "brams_3rd_order_5s_spectrogram.png"
    plt.savefig(img_filename, dpi=300, facecolor='black', edgecolor='none')
    print(f"Saved spectrogram plot to '{img_filename}'")

    plt.show()


if __name__ == "__main__":
    main()