import numpy as np
import scipy.io.wavfile as wav

# Ensure GUI backend works on macOS/Conda environments
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


def g3plx_meteor_simulation(
    sample_rate=44100,
    duration=60.0,
    f_carrier=1000.0,      # Audio center frequency (representing 49.97 MHz)
    snr_db=20.0
):
    """
    Implements Peter Martinez (G3PLX) model for wind-sheared meteor trail reflections.
    - Discretizes plasma column in 3D space
    - Applies altitude-dependent wind shear vector field
    - Calculates specular reflection points along the trail profile
    - Integrates complex phasor components (I/Q summation)
    """
    num_samples = int(sample_rate * duration)
    dt = 1.0 / sample_rate
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # --- 1. Trail Physics & Geometry ---
    t_entry = 2.0  # Meteor entry time (seconds)
    active_mask = t >= t_entry
    t_active = np.maximum(0.0, t - t_entry)

    # Spatial trail resolution along altitude axis z (-1.0 to 1.0 relative units)
    nz = 3500
    z = np.linspace(-1.2, 1.2, nz)
    dz = z[1] - z[0]

    # G3PLX Wind Shear Field V_wind(z): Multi-layer shear profile
    # Defines line-of-sight Doppler velocity per altitude layer (Hz)
    v_doppler_z = 85.0 * (np.sin(np.pi * z) + 0.35 * np.sin(3.0 * np.pi * z))

    # Initial trail tangent slope dx0/dz (nearly straight cylinder at impact)
    dx0_dz = np.gradient(0.015 * z, dz)
    
    # Shear gradient rate dV/dz (stretches plasma tube into serpentine shape)
    dv_dz = np.gradient(v_doppler_z, dz)

    # --- 2. Overdense Plasma Lifetimes ---
    # Fast initial formation (~0.05s) and slow electron recombination decay
    rise = 1.0 - np.exp(-t_active / 0.05)
    decay = np.exp(-t_active / 40.0)
    envelope = np.zeros(num_samples)
    envelope[active_mask] = rise[active_mask] * decay[active_mask]

    # --- 3. Initial Trunk Delay & Wind Shear Onset ---
    # Holds trail unbranched for ~0.5s (initial straight trail entry trunk)
    shear_onset_delay = 0.50
    shear_activation = 1.0 / (1.0 + np.exp(-(t_active - shear_onset_delay) / 0.08))

    # --- 4. G3PLX Complex Phasor Integration ---
    phase_z = np.zeros(nz)
    I_signal = np.zeros(num_samples)
    Q_signal = np.zeros(num_samples)

    for i in range(num_samples):
        if not active_mask[i]:
            continue

        tau = t_active[i]

        # Time-dependent shear slope dx/dz(t)
        # Asymptotic growth factor keeps specular points locked for 40-60+ seconds
        shear_factor = 0.14 * (1.0 - np.exp(-tau / 30.0)) * shear_activation[i]
        dxdz_t = dx0_dz + (dv_dz * shear_factor)

        # G3PLX Specular Condition Search:
        # Reflection occurs where trail tangent is perpendicular to radar vector (dx/dz ≈ 0)
        # Plasma diffusion widens reflection aperture over time
        aperture_width = 0.018 + (0.0004 * tau)
        specular_weight = np.exp(-(dxdz_t ** 2) / aperture_width)

        # Phase accumulation for each altitude layer
        inst_freq_z = f_carrier + v_doppler_z
        phase_z += 2.0 * np.pi * inst_freq_z * dt

        # Complex phasor summation (In-phase & Quadrature integration across trail length)
        I_signal[i] = np.sum(specular_weight * np.cos(phase_z)) * envelope[i]
        Q_signal[i] = np.sum(specular_weight * np.sin(phase_z)) * envelope[i]

    # Real RF audio signal from complex quadrature components
    echo_audio = I_signal

    # Normalize echo amplitude
    if np.max(np.abs(echo_audio)) > 0:
        echo_audio = echo_audio / np.max(np.abs(echo_audio))

    # --- 5. Direct Beacon Carrier Line (Narrow central reference at 0 Hz) ---
    beacon_signal = 0.035 * np.sin(2.0 * np.pi * f_carrier * t)

    # --- 6. Additive White Gaussian Noise (AWGN) ---
    total_signal = echo_audio + beacon_signal
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std = np.sqrt(np.mean(total_signal**2) / snr_linear)
    noise = np.random.normal(0, noise_std, num_samples)

    final_audio = total_signal + noise
    final_audio = final_audio / np.max(np.abs(final_audio))

    return t, final_audio


def main():
    SAMPLE_RATE = 44100
    DURATION = 60.0
    CARRIER_FREQ = 1000.0  # Represents 49.970 MHz BRAMS beacon

    print("Executing G3PLX Meteor Reflection Simulation...")
    t, audio = g3plx_meteor_simulation(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        f_carrier=CARRIER_FREQ,
        snr_db=18.0
    )

    # 1. Save WAV Audio
    wav_file = "g3plx_meteor_simulation.wav"
    wav.write(wav_file, SAMPLE_RATE, np.int16(audio * 32767))
    print(f"Saved simulation audio to '{wav_file}'")

    # 2. Audio Playback (If sounddevice is installed)
    if HAS_SOUNDDEVICE:
        print("Playing back simulation audio...")
        sd.play(np.int16(audio * 32767), SAMPLE_RATE)

    # 3. Save Spectrogram Image
    plt.figure(figsize=(14, 6), facecolor='black')
    ax = plt.axes()
    ax.set_facecolor('navy')

    # Spectrogram parameters tuned for fine line resolution
    plt.specgram(
        audio,
        NFFT=8192,
        Fs=SAMPLE_RATE,
        noverlap=7168,
        cmap='jet',
        vmin=-60,
        vmax=-10
    )

    plt.title("G3PLX Model — C-Shape / Epsilon Meteor Reflection Simulation", color='white', fontsize=12)
    plt.xlabel("Time (seconds)", color='white')
    plt.ylabel("Doppler Offset (Hz) [Relative to 49.970 MHz Beacon]", color='white')
    plt.ylim(CARRIER_FREQ - 100, CARRIER_FREQ + 100)

    # Custom Doppler Ticks (-100 Hz to +100 Hz)
    ticks = np.linspace(CARRIER_FREQ - 100, CARRIER_FREQ + 100, 9)
    tick_labels = [f"{int(f - CARRIER_FREQ):+d} Hz" for f in ticks]
    plt.yticks(ticks, tick_labels, color='white')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.grid(True, color='cyan', alpha=0.2, linestyle='--')
    plt.tight_layout()

    # Save PNG file explicitly to ensure visual output on all systems
    img_filename = "g3plx_spectrogram.png"
    plt.savefig(img_filename, dpi=300, facecolor='black', edgecolor='none')
    print(f"Saved spectrogram plot to '{img_filename}'")

    plt.show()


if __name__ == "__main__":
    main()