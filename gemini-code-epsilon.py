import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


def generate_epsilon_echo(
    sample_rate=44100,
    duration=6.0,          # Longer time duration for overdense trails
    ping_start=0.5,
    carrier_freq=800.0,    # CW carrier / base audio tone (Hz)
    num_branches=4,        # Number of trail fragments created by shear winds
    snr_db=20.0
):
    """
    Simulates a long-lasting overdense meteor echo with wind-shear distortion
    producing characteristic 'epsilon' or 'C-shaped' branches in the spectrogram.
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Active time array after entry
    active_mask = t >= ping_start
    t_act = np.zeros_like(t)
    t_act[active_mask] = t[active_mask] - ping_start
    
    # --- 1. Head Echo (Initial High-Speed Impact) ---
    # Fast frequency sweep downwards (Doppler shift of approaching meteor head)
    head_duration = 0.08  # ~80 ms head transit
    head_mask = active_mask & (t_act <= head_duration)
    
    phase_head = np.zeros_like(t)
    head_envelope = np.zeros_like(t)
    
    if np.any(head_mask):
        t_h = t_act[head_mask]
        # Linear/quadratic frequency drop representing high-speed approach
        f_start_offset = 120.0  # +120 Hz initial Doppler
        f_head = carrier_freq + f_start_offset * (1.0 - (t_h / head_duration)**2)
        phase_head[head_mask] = 2 * np.pi * np.cumsum(f_head) / sample_rate
        head_envelope[head_mask] = 0.8 * np.sin(np.pi * t_h / head_duration)

    # --- 2. Overdense Trail Decay & Multi-Branch Distortion ---
    # Overdense main trail uses a slower decay law: I(t) ~ sqrt(1 + t/tau) * exp(...)
    trail_tau = 1.8  # Longer lifetime (seconds)
    main_envelope = np.zeros_like(t)
    main_envelope[active_mask] = np.exp(-t_act[active_mask] / trail_tau)
    
    # Initialize composite signal accumulator
    composite_signal = head_envelope * np.sin(phase_head)
    
    # Generate random shear wind branches that create the "Epsilon" arms
    np.random.seed(42)  # Fixed seed for repeatable Epsilon shape
    
    for i in range(num_branches):
        # Each wind-shear branch turns on with a slight delay as trail deforms
        branch_delay = 0.1 + i * 0.35
        b_mask = active_mask & (t_act > branch_delay)
        
        if not np.any(b_mask):
            continue
            
        tb = t_act[b_mask] - branch_delay
        
        # C-shaped / Epsilon curve: Doppler shifts up/down due to turbulent wind vortex
        # Doppler curve combines a slow drift + sinusoidal oscillation
        drift_rate = np.random.choice([-1, 1]) * (8.0 + 5.0 * i)
        vortex_freq = 0.3 + 0.2 * i
        vortex_amp = 15.0 + 10.0 * i
        
        doppler_branch = (drift_rate * (1.0 - np.exp(-tb / 1.5)) + 
                          vortex_amp * np.sin(2 * np.pi * vortex_freq * tb))
        
        # Branch amplitude profile
        b_attack = 1.0 - np.exp(-tb / 0.15)
        b_decay = np.exp(-tb / (trail_tau * np.random.uniform(0.6, 1.0)))
        b_envelope = b_attack * b_decay * 0.6
        
        # Instantaneous phase calculation
        inst_freq = carrier_freq + doppler_branch
        phase_branch = 2 * np.pi * np.cumsum(inst_freq) / sample_rate
        
        # Add branch to full signal
        branch_sig = np.zeros_like(t)
        branch_sig[b_mask] = b_envelope * np.sin(phase_branch)
        composite_signal += branch_sig

    # --- 3. Add Background AWGN Noise ---
    sig_power = np.mean(composite_signal**2) + 1e-8
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_lin
    noise = np.random.normal(0, np.sqrt(noise_power), num_samples)
    
    audio = composite_signal + noise
    audio = audio / np.max(np.abs(audio))  # Normalize
    
    return t, audio


def main():
    SAMPLE_RATE = 44100
    DURATION = 5.5
    
    print("Generating Overdense 'Epsilon' Meteor Echo...")
    t, audio = generate_epsilon_echo(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        ping_start=0.5,
        carrier_freq=900.0,
        num_branches=5,
        snr_db=18.0
    )

    # Save to WAV file
    wav_filename = "meteor_epsilon_echo.wav"
    audio_int16 = np.int16(audio * 32767)
    wav.write(wav_filename, SAMPLE_RATE, audio_int16)
    print(f"Saved simulation audio to '{wav_filename}'")

    if HAS_SOUNDDEVICE:
        print("Playing audio...")
        sd.play(audio_int16, SAMPLE_RATE)
        sd.wait()

    # Plot Spectrogram (Crucial for visualizing Epsilon shapes)
    plt.figure(figsize=(11, 6))
    
    # NFFT size tuned for high time-frequency compromise
    plt.specgram(
        audio, 
        NFFT=2048, 
        Fs=SAMPLE_RATE, 
        noverlap=1792, 
        cmap='inferno',
        vmin=-80  # dB dynamic range limit
    )
    
    plt.title("Meteor Scatter Spectrogram — Overdense 'Epsilon' Echo (G3PLX Model)", fontsize=12)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(700, 1100)  # Zoom in on carrier region
    plt.colorbar(label='Power (dB)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()