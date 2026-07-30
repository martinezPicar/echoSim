import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

def generate_g3plx_audio(mode='epsilon', duration=5.0, sample_rate=44100, carrier_freq=1500.0):
    num_samples = int(sample_rate * duration)
    dt = 1.0 / sample_rate
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    z = np.linspace(-1.0, 1.0, 1000)
    dz = z[1] - z[0]
    
    if mode == 'C_shape':
        x = np.sin(np.pi * z)
    else:  # 'epsilon'
        x = np.sin(np.pi * z) + 0.35 * np.sin(3.0 * np.pi * z)
        
    dxdz = np.gradient(x, dz)
    
    t_center = duration / 2.0
    rotation_angle = 1.0 * (t - t_center)
    
    audio = np.zeros(num_samples)
    
    # Track phase for active specular tracks
    max_tracks = 4
    phases = np.zeros(max_tracks)
    
    for i in range(num_samples):
        angle = rotation_angle[i]
        tan_angle = np.tan(angle)
        
        # Find discrete local minima (specular points)
        diff = np.abs(dxdz - tan_angle)
        local_mins = []
        for j in range(1, len(diff) - 1):
            if diff[j] < diff[j-1] and diff[j] < diff[j+1] and diff[j] < 0.1:
                local_mins.append(j)
        
        sample_val = 0.0
        for track_idx, m_idx in enumerate(local_mins[:max_tracks]):
            doppler = 20.0 * x[m_idx] * np.cos(angle)
            inst_f = carrier_freq + doppler
            
            phases[track_idx] += 2.0 * np.pi * inst_f * dt
            sample_val += np.sin(phases[track_idx])
            
        # Global amplitude envelope
        envelope = np.exp(-((t[i] - t_center)**2) / (2 * 1.0**2))
        audio[i] = sample_val * envelope

    # Normalize & add minor noise
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    noise = np.random.normal(0, 0.02, num_samples)
    final_audio = audio + noise
    
    return t, final_audio / np.max(np.abs(final_audio))

# Test audio generation and plot high-res spectrogram
sample_rate = 44100
t, audio = generate_g3plx_audio(mode='epsilon', duration=5.0, sample_rate=sample_rate)

# Save WAV
wav.write("g3plx_epsilon.wav", sample_rate, np.int16(audio * 32767))

# Spectrogram
plt.figure(figsize=(10, 5))
plt.specgram(audio, NFFT=4096, Fs=sample_rate, noverlap=3840, cmap='inferno', vmin=-60)
plt.title("G3PLX 'Epsilon' Audio Spectrogram")
plt.ylim(1460, 1540)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="dB")
plt.tight_layout()
plt.show()