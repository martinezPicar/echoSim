import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

def g3plx_meteor_simulator(
    sample_rate=44100,
    duration=6.0,
    carrier_freq=1500.0,   # Base audio frequency (Hz)
    mode='epsilon',        # 'C_shape' or 'epsilon'
    snr_db=25.0
):
    """
    Simulates G3PLX meteor echoes by finding continuous specular reflection 
    trajectories along a wind-deformed trail.
    """
    num_samples = int(sample_rate * duration)
    dt = 1.0 / sample_rate
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # 1. Define Trail Geometry (z along length in [-1, 1])
    nz = 1000
    z = np.linspace(-1.0, 1.0, nz)
    dz = z[1] - z[0]
    
    # Wind shear displacement profile x(z)
    if mode == 'C_shape':
        # 1st order sine distortion -> C / U shape
        x = np.sin(np.pi * z)
    else:  # 'epsilon'
        # Fundamental + 3rd order harmonic -> Epsilon / W shape
        x = np.sin(np.pi * z) + 0.45 * np.sin(3.0 * np.pi * z)
        
    x = x / np.max(np.abs(x))  # Normalize displacement
    
    # Derivative dx/dz (local tangent slope of the trail)
    dxdz = np.gradient(x, dz)
    
    # 2. Time-evolution parameters
    # The trail rotates through the specular condition angle
    t_center = duration / 2.0
    rotation_speed = 0.8  # Speed of trail rotation / wind drift
    
    # Angle theta of trail relative to radar aspect angle over time
    theta_t = rotation_speed * (t - t_center)
    
    # Signal synthesis via instantaneous phase integration
    signal = np.zeros(num_samples)
    current_phase = np.zeros(nz)
    
    # Doppler scaling factor (Hz)
    max_doppler_hz = 25.0 
    
    for i in range(num_samples):
        tan_theta = np.tan(theta_t[i])
        
        # Specular reflection condition: dx/dz == tan(theta)
        # We calculate specular weight with a Gaussian aperture window
        specular_weight = np.exp(-((dxdz - tan_theta) ** 2) / 0.02)
        
        # Calculate local Doppler offset along trail profile
        # Doppler shift is proportional to local velocity towards observer
        doppler_z = max_doppler_hz * x * np.cos(theta_t[i])
        
        # Update continuous phases per point to avoid phase discontinuity jumps
        inst_freq = carrier_freq + doppler_z
        current_phase += 2.0 * np.pi * inst_freq * dt
        
        # Trail envelope: turns on near alignment, decays over time
        alignment_factor = np.mean(specular_weight)
        decay = np.exp(-((t[i] - t_center) ** 2) / (2.0 * 1.2 ** 2))
        
        # Sum reflections across specular zones
        reflections = specular_weight * np.sin(current_phase)
        signal[i] = np.sum(reflections) * decay

    # 3. Additive Noise & Normalization
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
        
    snr_lin = 10 ** (snr_db / 10.0)
    noise = np.random.normal(0, 1.0 / np.sqrt(snr_lin), num_samples)
    
    audio = signal + noise
    audio = audio / np.max(np.abs(audio))
    return t, audio

def main():
    SAMPLE_RATE = 44100
    DURATION = 5.0
    MODE = 'epsilon'  # Change to 'C_shape' or 'epsilon'
    
    print(f"Generating G3PLX '{MODE}' trace...")
    t, audio = g3plx_meteor_simulator(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        carrier_freq=1500.0,
        mode=MODE,
        snr_db=30.0
    )

    # Save output to WAV
    wav_file = f"g3plx_clean_{MODE}.wav"
    wav.write(wav_file, SAMPLE_RATE, np.int16(audio * 32767))
    print(f"Saved audio to '{wav_file}'")

    if HAS_SOUNDDEVICE:
        sd.play(np.int16(audio * 32767), SAMPLE_RATE)

    # High-resolution spectrogram matching SpectrumLab style
    plt.figure(figsize=(11, 5))
    
    # High time-frequency resolution windowing
    nfft = 4096
    overlap = 3840
    
    plt.specgram(
        audio, 
        NFFT=nfft, 
        Fs=SAMPLE_RATE, 
        noverlap=overlap, 
        cmap='inferno',
        vmin=-80
    )
    
    plt.title(f"G3PLX Meteor Simulation Spectrogram — '{MODE}' Echo", fontsize=12)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(1460, 1540)  # Zoom in on the 1500 Hz tone region
    plt.colorbar(label='Power (dB)')
    plt.grid(True, alpha=0.2, color='white')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()