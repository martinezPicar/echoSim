import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


def generate_brams_epsilon_audio(
    duration=10.0,
    sample_rate=44100,
    carrier_freq=1000.0,
    snr_db=15.0
):
    """
    Generates time-domain audio simulating the multi-loop BRAMS wind-shear 
    'epsilon / serpentine' meteor echo based on G3PLX's specular model.
    """
    num_samples = int(sample_rate * duration)
    dt = 1.0 / sample_rate
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Trail event timing (start at 6.0s, lasts ~3.5 seconds)
    echo_start = 5.5
    echo_duration = 3.5
    
    # Active mask
    active_mask = (t >= echo_start) & (t <= (echo_start + echo_duration))
    t_active = t - echo_start

    # Continuous phase accumulator
    phase_sum = np.zeros(num_samples)
    audio_signal = np.zeros(num_samples)

    # Trail Geometry & Wind Shear Parameters
    # Sinusoidal wind-shear distortion frequency (creates stacked loops)
    shear_cycles = 2.5       # Number of C/Epsilon loops stacked vertically
    doppler_amplitude = 35.0  # Doppler spread (+/- Hz)
    drift_rate = 0.8         # Rotation speed through specular alignment

    # Calculate instantaneous Doppler curve matching the BRAMS geometry
    for i in range(num_samples):
        if active_mask[i]:
            tau = t_active[i]
            
            # Rotation angle passing through radar alignment
            angle = drift_rate * (tau - (echo_duration / 2.0))
            
            # Multi-harmonic wind shear function (1st + 3rd order for Epsilon loops)
            shear_profile = (np.sin(2 * np.pi * shear_cycles * (tau / echo_duration)) + 
                             0.3 * np.cos(6 * np.pi * shear_cycles * (tau / echo_duration)))
            
            # Local Doppler shift along specular reflection zone
            doppler_hz = doppler_amplitude * shear_profile * np.cos(angle)
            
            # Smooth envelope (fast attack, slow diffusion decay)
            envelope = np.sin(np.pi * (tau / echo_duration)) ** 0.5
            
            # Accumulate continuous phase
            inst_freq = carrier_freq + doppler_hz
            if i > 0:
                phase_sum[i] = phase_sum[i-1] + 2 * np.pi * inst_freq * dt
            else:
                phase_sum[i] = 2 * np.pi * inst_freq * dt
                
            audio_signal[i] = envelope * np.sin(phase_sum[i])

    # Additive White Gaussian Noise (AWGN) matching background RF noise
    signal_power = np.mean(audio_signal[active_mask]**2) if np.any(active_mask) else 1.0
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_lin
    noise = np.random.normal(0, np.sqrt(noise_power), num_samples)

    # Combine & Normalize
    audio = audio_signal + noise
    audio = audio / np.max(np.abs(audio))
    return t, audio


def main():
    SAMPLE_RATE = 44100
    DURATION = 10.0
    CARRIER_FREQ = 1000.0  # Audio tone center frequency (Hz)

    print("Generating BRAMS-style Epsilon Meteor Echo...")
    t, audio = generate_brams_epsilon_audio(
        duration=DURATION,
        sample_rate=SAMPLE_RATE,
        carrier_freq=CARRIER_FREQ,
        snr_db=18.0
    )

    # Save to WAV
    wav_filename = "brams_epsilon_sim.wav"
    wav.write(wav_filename, SAMPLE_RATE, np.int16(audio * 32767))
    print(f"Saved audio output to '{wav_filename}'")

    if HAS_SOUNDDEVICE:
        print("Playing audio output...")
        sd.play(np.int16(audio * 32767), SAMPLE_RATE)

    # Plot using BRAMS Blue/Jet Spectrogram styling
    plt.figure(figsize=(10, 6), facecolor='black')
    ax = plt.axes()
    ax.set_facecolor('navy')

    # Spectrogram parameters tuned for high resolution on narrow bands
    plt.specgram(
        audio,
        NFFT=2048,
        Fs=SAMPLE_RATE,
        noverlap=1920,
        cmap='jet',       # Classic BRAMS blue-to-red color map
        vmin=-65,
        vmax=-10
    )

    plt.title("BRAMS Meteor Network — Simulated Epsilon Echo", color='white', fontsize=12)
    plt.xlabel("Time (seconds)", color='white')
    plt.ylabel("Frequency (Hz)", color='white')
    
    # Zoom into the carrier tone frequency window
    plt.ylim(CARRIER_FREQ - 60, CARRIER_FREQ + 60)
    
    # Format dark-mode axes matching BRAMS displays
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.grid(True, color='cyan', alpha=0.15)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()