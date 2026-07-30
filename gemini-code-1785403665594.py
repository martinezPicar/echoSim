import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

def generate_meteor_ping(
    sample_rate=44100,      # Audio sample rate (Hz)
    duration=3.0,           # Total simulation time (seconds)
    ping_start=0.5,         # Time when the meteor enters (seconds)
    carrier_freq=1000.0,    # Audio tone pitch / CW frequency (Hz)
    doppler_shift=15.0,     # Initial Doppler offset (Hz)
    decay_tau=0.25,         # Decay constant (seconds) - typical underdense ping
    fresnel_freq=12.0,      # Fresnel ripple frequency during trail formation
    snr_db=15.0             # Signal-to-Noise Ratio (dB) peak
):
    """
    Simulates a meteor scatter RF/audio 'ping' based on the underdense 
    ionized trail reflection model.
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # 1. Trail Envelope Construction
    envelope = np.zeros_like(t)
    active_mask = t >= ping_start
    t_active = t[active_mask] - ping_start
    
    # Attack profile (~5 ms rise time)
    attack_tau = 0.005
    attack = 1.0 - np.exp(-t_active / attack_tau)
    
    # Decay profile (Exponential electron diffusion)
    decay = np.exp(-t_active / decay_tau)
    
    # Optional: Fresnel zone amplitude oscillation during formation
    fresnel_ripple = 1.0 + 0.15 * np.cos(2 * np.pi * fresnel_freq * np.sqrt(t_active + 1e-5))
    
    envelope[active_mask] = attack * decay * fresnel_ripple

    # 2. Carrier & Doppler Modulation
    # Exponentially decaying Doppler shift as trail decelerates/stabilizes
    doppler_t = doppler_shift * np.exp(-t_active / (decay_tau * 2)) if np.any(active_mask) else 0
    phase_doppler = np.zeros_like(t)
    phase_doppler[active_mask] = 2 * np.pi * doppler_t * t_active
    
    # Base RF audio carrier
    phase_carrier = 2 * np.pi * carrier_freq * t
    signal = envelope * np.sin(phase_carrier + phase_doppler)

    # 3. Additive White Gaussian Noise (AWGN)
    signal_power = np.mean(envelope**2) if np.max(envelope) > 0 else 1e-6
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), num_samples)

    # Final combined audio wave
    audio = signal + noise
    
    # Normalize to prevents clipping (-1.0 to 1.0 range)
    audio = audio / np.max(np.abs(audio))
    return t, audio, envelope

def main():
    # Simulation Parameters
    SAMPLE_RATE = 44100
    DURATION = 3.0
    
    print("Generating Meteor Scatter Ping...")
    t, audio, envelope = generate_meteor_ping(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        ping_start=0.4,
        carrier_freq=800.0,   # 800 Hz CW ping
        doppler_shift=20.0,   # 20 Hz initial Doppler pitch drop
        decay_tau=0.20,       # 200 ms trail lifetime
        snr_db=18.0           # 18 dB peak SNR
    )

    # Save to WAV file
    wav_filename = "meteor_ping.wav"
    audio_int16 = np.int16(audio * 32767)
    wav.write(wav_filename, SAMPLE_RATE, audio_int16)
    print(f"Saved audio simulation to '{wav_filename}'")

    # Play back audio if sounddevice is available
    if HAS_SOUNDDEVICE:
        print("Playing audio...")
        sd.play(audio_int16, SAMPLE_RATE)
        sd.wait()

    # Plot Signal Waveform and Spectrogram
    plt.figure(figsize=(10, 6))

    # Subplot 1: Time domain waveform
    plt.subplot(2, 1, 1)
    plt.plot(t, audio, color='gray', alpha=0.6, label='Audio + Noise')
    plt.plot(t, envelope, color='red', linewidth=1.5, label='Meteor Trail Envelope')
    plt.title("Meteor Scatter Ping Simulation (Time Domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True)

    # Subplot 2: Spectrogram
    plt.subplot(2, 1, 2)
    plt.specgram(audio, NFFT=1024, Fs=SAMPLE_RATE, noverlap=512, cmap='inferno')
    plt.title("Spectrogram (Doppler & Frequency Burst)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(400, 1200)  # Zoom in on carrier tone
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()