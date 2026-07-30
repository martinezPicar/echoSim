import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


def generate_extended_initial_trunk_epsilon(
    sample_rate=44100,
    duration=60.0,        # Total display window
    carrier_freq=1000.0,   # Represents 49.970 MHz BRAMS carrier
    snr_db=18.0
):
    """
    Simulates a long-duration BRAMS overdense meteor echo with an extended initial 
    unbranched trunk (~0.35s) before splitting into long-lasting persistent branches.
    """
    num_samples = int(sample_rate * duration)
    dt = 1.0 / sample_rate
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # --- Timing ---
    t_entry = 2.0  # Meteor impact at t = 2.0s
    active_mask = t >= t_entry
    t_active = np.maximum(0.0, t - t_entry)

    # --- Overdense Plasma Envelope ---
    # Fast attack (~0.05s) and slow diffusion decay over 60s
    rise_envelope = 1.0 - np.exp(-t_active / 0.05)
    decay_envelope = np.exp(-t_active / 35.0)
    global_envelope = np.zeros(num_samples)
    global_envelope[active_mask] = rise_envelope[active_mask] * decay_envelope[active_mask]

    # --- Trail Spatial Discretization ---
    nz = 4000
    z = np.linspace(-1.2, 1.2, nz)
    dz = z[1] - z[0]

    # Multi-layer wind shear velocity profile (Doppler Hz shift per altitude layer)
    v_wind = 80.0 * (np.sin(np.pi * z) + 0.4 * np.sin(3.0 * np.pi * z))

    # Initial flat slope profile (produces single central specular reflection at entry)
    dx0_dz = np.gradient(0.02 * z, dz)
    
    # Shear gradient rate dV/dz
    dv_dz = np.gradient(v_wind, dz)

    phase_z = np.zeros(nz)
    echo_signal = np.zeros(num_samples)

    # --- Shear Delay Parameter ---
    # Holds trail unbranched for ~0.35s after impact
    shear_onset_delay = 0.35  

    for i in range(num_samples):
        if not active_mask[i]:
            continue

        tau = t_active[i]

        # Sigmoidal shear activation: delays branch formation for ~0.35s
        # Before tau = 0.35s, shear_activation ~ 0 (trail stays straight & unbranched)
        # After tau = 0.35s, branches smoothly emerge and persist
        shear_activation = 1.0 / (1.0 + np.exp(-(tau - shear_onset_delay) / 0.08))
        
        # Slow asymptotic shear evolution so branches last 40-50+ seconds
        shear_factor = 0.12 * (1.0 - np.exp(-tau / 25.0)) * shear_activation
        dxdz_t = dx0_dz + (dv_dz * shear_factor)

        # Broad specular reflection aperture keeps branches strong and visible long-term
        aperture_width = 0.02 + (0.0005 * tau)
        specular_weight = np.exp(-(dxdz_t ** 2) / aperture_width)

        # Local Doppler shift along altitude
        doppler_z = v_wind

        # Continuous phase accumulation
        inst_freq_z = carrier_freq + doppler_z
        phase_z += 2.0 * np.pi * inst_freq_z * dt

        # Sum reflections across all specular regions
        reflections = specular_weight * np.sin(phase_z)
        echo_signal[i] = np.sum(reflections) * global_envelope[i]

    # Normalize echo signal strength
    if np.max(np.abs(echo_signal)) > 0:
        echo_signal = echo_signal / np.max(np.abs(echo_signal))

    # Faint direct BRAMS beacon signal at 0 Hz offset
    beacon_signal = 0.04 * np.sin(2.0 * np.pi * carrier_freq * t)

    # Combine Echo + Direct Beacon + Background AWGN
    total_sig = echo_signal + beacon_signal
    snr_lin = 10 ** (snr_db / 10.0)
    noise = np.random.normal(0, 1.0 / np.sqrt(snr_lin), num_samples)

    audio = total_sig + noise
    return t, audio / np.max(np.abs(audio))


def main():
    SAMPLE_RATE = 44100
    DURATION = 60.0
    CARRIER_FREQ = 1000.0

    print("Generating BRAMS Epsilon Simulation with Extended Initial Trunk...")
    t, audio = generate_extended_initial_trunk_epsilon(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        carrier_freq=CARRIER_FREQ,
        snr_db=18.0
    )

    # Save output to WAV
    wav_file = "brams_extended_trunk_epsilon.wav"
    wav.write(wav_file, SAMPLE_RATE, np.int16(audio * 32767))
    print(f"Saved simulation audio to '{wav_file}'")

    if HAS_SOUNDDEVICE:
        sd.play(np.int16(audio * 32767), SAMPLE_RATE)

    # Spectrogram Display
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

    plt.title("BRAMS Network (49.97 MHz) — Overdense Echo with Extended Initial Trunk", color='white', fontsize=12)
    plt.xlabel("Time (seconds)", color='white')
    plt.ylabel("Doppler Offset (Hz) [Relative to 49.970 MHz Beacon]", color='white')
    plt.ylim(CARRIER_FREQ - 100, CARRIER_FREQ + 100)

    ticks = np.linspace(CARRIER_FREQ - 100, CARRIER_FREQ + 100, 9)
    tick_labels = [f"{int(f - CARRIER_FREQ):+d} Hz" for f in ticks]
    plt.yticks(ticks, tick_labels, color='white')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.grid(True, color='cyan', alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()