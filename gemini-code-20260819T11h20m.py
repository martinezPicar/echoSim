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


def simulate_physical_meteor_echo(
    sample_rate=44100,
    duration=60.0,
    carrier_freq=1000.0,  # Audio carrier representing BRAMS 49.970 MHz beacon
    snr_db=18.0
):
    """
    Simulates a 2-Phase Overdense Meteor Reflection:
    - Phase 1 (0 to 0.5s): Fast entry phase, forming an unbranched vertical trunk 
      with a narrow initial Doppler spread (±4.5 Hz).
    - Phase 2 (0.5s to 60s): Persistent wind-shear distortion, bending the trail 
      into sinusoidal curves that create approaching (+Hz) and receding (-Hz) branches.
    """
    num_samples = int(sample_rate * duration)
    dt = 1.0 / sample_rate
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # --- 1. Event Timing ---
    t_entry = 2.0  # Impact occurs at t = 2.0s
    active_mask = t >= t_entry
    t_active = np.maximum(0.0, t - t_entry)

    # --- 2. Overdense Plasma Decay Envelope ---
    # Fast formation (~0.02s) and slow electron diffusion decay (35s constant)
    rise_env = 1.0 - np.exp(-t_active / 0.02)
    decay_env = np.exp(-t_active / 35.0)
    global_envelope = np.zeros(num_samples)
    global_envelope[active_mask] = rise_env[active_mask] * decay_env[active_mask]

    # --- 3. Spatial Discretization of the Trail Column ---
    nz = 4000
    z = np.linspace(-1.2, 1.2, nz)
    dz = z[1] - z[0]

    # Initial straight cylinder slope (produces single specular reflection)
    dx0_dz = np.gradient(0.01 * z, dz)

    # Sinusoidal Wind Shear Profile (Line-of-sight Doppler velocity field)
    # Approaching (+Hz) and receding (-Hz) wind vectors across altitude z
    v_wind = 75.0 * (np.sin(np.pi * z) + 0.35 * np.sin(3.0 * np.pi * z))
    dv_dz = np.gradient(v_wind, dz)

    # --- 4. Signal Synthesis Loop ---
    phase_z = np.zeros(nz)
    echo_signal = np.zeros(num_samples)

    # Phase 1: Entry Head Doppler parameter (±4.5 Hz peak deviation)
    trunk_duration = 0.50  # Initial trunk lasts 0.5s

    for i in range(num_samples):
        if not active_mask[i]:
            continue

        tau = t_active[i]

        if tau < trunk_duration:
            # PHASE 1: Straight-trail entry phase (Initial Trunk)
            # Narrow Doppler spread constrained within ±4.5 Hz around carrier
            doppler_z = 4.5 * np.sin(2.0 * np.pi * z)
            specular_weight = np.exp(-(dx0_dz ** 2) / 0.01)

        else:
            # PHASE 2: Wind Shear Distortion Phase
            # Smooth transition activating the sinusoidal bending
            shear_tau = tau - trunk_duration
            shear_factor = 0.12 * (1.0 - np.exp(-shear_tau / 20.0))
            dxdz_t = dx0_dz + (dv_dz * shear_factor)

            # Expanding plasma radius broadens specular aperture over time
            aperture_width = 0.015 + (0.0004 * tau)
            specular_weight = np.exp(-(dxdz_t ** 2) / aperture_width)

            # Deformed trail Doppler offset (+/– Hz depending on wind direction)
            doppler_z = v_wind

        # Instantaneous phase accumulation per altitude layer
        inst_freq_z = carrier_freq + doppler_z
        phase_z += 2.0 * np.pi * inst_freq_z * dt

        # Sum complex reflections across all specular zones
        reflections = specular_weight * np.sin(phase_z)
        echo_signal[i] = np.sum(reflections) * global_envelope[i]

    # Normalize echo signal strength
    if np.max(np.abs(echo_signal)) > 0:
        echo_signal = echo_signal / np.max(np.abs(echo_signal))

    # --- 5. Direct Reference Beacon Carrier (Narrow 0 Hz Baseline) ---
    beacon_signal = 0.035 * np.sin(2.0 * np.pi * carrier_freq * t)

    # Combine Echo + Direct Beacon + Background Noise (AWGN)
    total_sig = echo_signal + beacon_signal
    snr_lin = 10 ** (snr_db / 10.0)
    noise = np.random.normal(0, np.sqrt(np.mean(total_sig**2) / snr_lin), num_samples)

    final_audio = total_sig + noise
    return t, final_audio / np.max(np.abs(final_audio))


def main():
    SAMPLE_RATE = 44100
    DURATION = 60.0
    CARRIER_FREQ = 1000.0

    print("Generating Physical 2-Phase Meteor Echo Simulation...")
    t, audio = simulate_physical_meteor_echo(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        carrier_freq=CARRIER_FREQ,
        snr_db=18.0
    )

    # 1. Save WAV Audio
    wav_file = "physical_meteor_echo.wav"
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

    plt.title("BRAMS Network — Physical 2-Phase Meteor Echo (±4.5 Hz Trunk → Sinusoidal Wind Shear)", color='white', fontsize=11)
    plt.xlabel("Time (seconds)", color='white')
    plt.ylabel("Doppler Offset (Hz) [Relative to Direct Beacon]", color='white')
    plt.ylim(CARRIER_FREQ - 100, CARRIER_FREQ + 100)

    # Custom Y-Axis Labels (-100 Hz to +100 Hz offset)
    ticks = np.linspace(CARRIER_FREQ - 100, CARRIER_FREQ + 100, 9)
    tick_labels = [f"{int(f - CARRIER_FREQ):+d} Hz" for f in ticks]
    plt.yticks(ticks, tick_labels, color='white')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.grid(True, color='cyan', alpha=0.18, linestyle='--')
    plt.tight_layout()

    # Save PNG directly to disk to ensure GUI independence
    img_filename = "physical_meteor_spectrogram.png"
    plt.savefig(img_filename, dpi=300, facecolor='black', edgecolor='none')
    print(f"Saved spectrogram plot to '{img_filename}'")

    plt.show()


if __name__ == "__main__":
    main()