#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import wave
import struct

class EpsilonMeteorSimulator:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
    def generate_epsilon_echo(self, duration=10.0, center_freq=1000, 
                            max_doppler=50, noise_level=0.05, 
                            turbulence_level=0.1, echo_strength=0.8):
        """
        Generate epsilon meteor echo simulation
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Epsilon echo characteristics
        build_up = 0.5  # seconds to reach peak
        sustain = 2.0   # seconds of sustained echo
        decay = duration - build_up - sustain
        
        amplitude = np.zeros_like(t)
        amplitude[t < build_up] = t[t < build_up] / build_up
        amplitude[(t >= build_up) & (t < build_up + sustain)] = 1.0
        decay_time = t[t >= build_up + sustain] - (build_up + sustain)
        amplitude[t >= build_up + sustain] = np.exp(-decay_time / (decay * 0.5))
        
        # Complex Doppler evolution
        doppler = max_doppler * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)) * np.exp(-t / (duration * 0.7))
        
        # Phase fluctuations
        phase_noise = turbulence_level * np.cumsum(np.random.randn(len(t))) / self.sample_rate
        
        # Generate the signal
        carrier = np.sin(2 * np.pi * center_freq * t + 
                        2 * np.pi * np.cumsum(doppler) / self.sample_rate +
                        2 * np.pi * phase_noise)
        
        signal = echo_strength * amplitude * carrier
        
        # Add noise
        noise = noise_level * np.random.randn(len(t))
        signal += noise
        
        # Normalize
        signal = 0.9 * signal / np.max(np.abs(signal))
        
        return signal, t, amplitude, doppler
    
    def save_wav(self, signal, filename, sample_rate=44100):
        """Save signal as WAV file using standard wave module"""
        # Convert to 16-bit PCM
        signal_int = np.int16(signal * 32767)
        
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes = 16 bits
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(signal_int.tobytes())
        
        print(f"Saved {filename}")
    
    def plot_echo_characteristics(self, t, signal, amplitude, doppler):
        """Plot the epsilon echo characteristics"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
        
        ax1.plot(t, signal, 'b-', alpha=0.7)
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Epsilon Meteor Echo - Time Domain')
        ax1.grid(True)
        
        ax2.plot(t, amplitude, 'r-', linewidth=2)
        ax2.set_ylabel('Normalized Amplitude')
        ax2.set_title('Amplitude Envelope')
        ax2.grid(True)
        
        ax3.plot(t, doppler, 'g-', linewidth=2)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Doppler Shift (Hz)')
        ax3.set_title('Doppler Evolution')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig('epsilon_echo_characteristics.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_multiple_echoes(self, num_echoes=5, min_duration=3.0, 
                               max_duration=15.0, **kwargs):
        """Generate multiple epsilon echoes"""
        all_signals = []
        metadata = []
        
        for i in range(num_echoes):
            duration = np.random.uniform(min_duration, max_duration)
            center_freq = kwargs.get('center_freq', 800 + np.random.uniform(-100, 100))
            max_doppler = kwargs.get('max_doppler', 20 + np.random.uniform(0, 30))
            
            sig, t, amp, dopp = self.generate_epsilon_echo(
                duration=duration,
                center_freq=center_freq,
                max_doppler=max_doppler,
                noise_level=kwargs.get('noise_level', 0.05),
                turbulence_level=kwargs.get('turbulence_level', 0.1),
                echo_strength=kwargs.get('echo_strength', 0.7 + np.random.uniform(-0.2, 0.2))
            )
            
            all_signals.append(sig)
            metadata.append({
                'duration': duration,
                'center_freq': center_freq,
                'max_doppler': max_doppler,
                'amplitude_envelope': amp,
                'doppler_profile': dopp
            })
        
        return all_signals, metadata

def main():
    # Initialize simulator
    simulator = EpsilonMeteorSimulator(sample_rate=44100)
    
    # Generate single epsilon echo
    print("Generating epsilon meteor echo...")
    signal, t, amplitude, doppler = simulator.generate_epsilon_echo(
        duration=12.0,
        center_freq=800,
        max_doppler=40,
        noise_level=0.03,
        turbulence_level=0.08,
        echo_strength=0.85
    )
    
    # Save audio
    simulator.save_wav(signal, 'epsilon_meteor_echo.wav')
    
    # Plot characteristics
    simulator.plot_echo_characteristics(t, signal, amplitude, doppler)
    
    # Generate multiple echoes
    print("\nGenerating multiple epsilon echoes...")
    all_echoes, metadata = simulator.generate_multiple_echoes(
        num_echoes=3,
        min_duration=8.0,
        max_duration=20.0
    )
    
    # Save multiple echoes
    for i, echo in enumerate(all_echoes):
        simulator.save_wav(echo, f'epsilon_echo_{i+1}.wav')
        print(f"Echo {i+1}: duration={metadata[i]['duration']:.1f}s, "
              f"freq={metadata[i]['center_freq']:.0f}Hz, "
              f"doppler={metadata[i]['max_doppler']:.1f}Hz")
    
    print("\nEpsilon meteor simulation complete!")

if __name__ == "__main__":
    main()
