#!/usr/bin/env python3
import numpy as np
import soundfile as sf
import argparse

def generate_meteor_ping(duration=1.0, sample_rate=44100, 
                        center_freq=1000, doppler_shift=200,
                        amplitude_decay=0.5, noise_level=0.1):
    """
    Generate a realistic meteor ping simulation
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Exponential amplitude decay (simulating trail dissipation)
    amplitude = np.exp(-t / amplitude_decay)
    
    # Time-varying Doppler shift (simulating trail evolution)
    doppler = doppler_shift * np.exp(-t / (duration * 0.5))
    
    # Generate the signal
    signal = amplitude * np.sin(2 * np.pi * (center_freq + doppler) * t)
    
    # Add noise
    noise = noise_level * np.random.randn(len(t))
    signal += noise
    
    # Normalize
    signal = 0.9 * signal / np.max(np.abs(signal))
    
    return signal, sample_rate

def main():
    parser = argparse.ArgumentParser(description='Meteor Ping Simulator')
    parser.add_argument('--duration', type=float, default=1.0, help='Ping duration in seconds')
    parser.add_argument('--output', type=str, default='meteor_ping.wav', help='Output filename')
    parser.add_argument('--freq', type=float, default=1000, help='Center frequency in Hz')
    parser.add_argument('--doppler', type=float, default=200, help='Maximum Doppler shift in Hz')
    
    args = parser.parse_args()
    
    signal, sr = generate_meteor_ping(
        duration=args.duration,
        center_freq=args.freq,
        doppler_shift=args.doppler
    )
    
    sf.write(args.output, signal, sr)
    print(f"Generated meteor ping: {args.output}")

if __name__ == "__main__":
    main()
