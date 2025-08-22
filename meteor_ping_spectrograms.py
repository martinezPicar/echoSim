#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse

def generate_single_ping(duration, sample_rate, center_freq, doppler_shift,
                         amplitude_decay, noise_level, start_time, total_duration):
    """
    Generate a single meteor ping event placed at a given start time
    """
    n_total = int(sample_rate * total_duration)
    signal = np.zeros(n_total)

    t = np.linspace(0, duration, int(sample_rate * duration))

    # Exponential amplitude decay (simulating trail dissipation)
    amplitude = np.exp(-t / amplitude_decay)

    # Time-varying Doppler shift (simulating trail evolution)
    doppler = doppler_shift * np.exp(-t / (duration * 0.5))

    # Generate the ping
    ping = amplitude * np.sin(2 * np.pi * (center_freq + doppler) * t)

    # Add noise
    ping += noise_level * np.random.randn(len(ping))

    # Normalize
    ping = 0.9 * ping / np.max(np.abs(ping))

    # Insert ping into overall signal
    start_idx = int(start_time * sample_rate)
    end_idx = min(start_idx + len(ping), n_total)
    signal[start_idx:end_idx] = ping[:end_idx - start_idx]

    return signal

def main():
    parser = argparse.ArgumentParser(description='Meteor Shower Simulator (Spectrogram output)')
    parser.add_argument('--duration', type=float, default=10.0, help='Total simulation duration in seconds')
    parser.add_argument('--output', type=str, default='meteor_shower.png', help='Output spectrogram filename')
    parser.add_argument('--freq', type=float, default=1000, help='Center frequency in Hz')
    parser.add_argument('--doppler', type=float, default=200, help='Maximum Doppler shift in Hz')
    parser.add_argument('--events', type=int, default=10, help='Number of meteor pings to simulate')
    
    args = parser.parse_args()
    sr = 44100
    total_signal = np.zeros(int(sr * args.duration))

    rng = np.random.default_rng()

    for _ in range(args.events):
        # Randomize each event
        ping_dur = rng.uniform(0.2, 1.0)                # seconds
        start_time = rng.uniform(0, args.duration - ping_dur)
        doppler = rng.uniform(-args.doppler, args.doppler)
        decay = rng.uniform(0.1, 0.8)                   # exponential decay constant
        noise_level = rng.uniform(0.02, 0.1)

        # Add ping to signal
        total_signal += generate_single_ping(
            duration=ping_dur,
            sample_rate=sr,
            center_freq=args.freq,
            doppler_shift=doppler,
            amplitude_decay=decay,
            noise_level=noise_level,
            start_time=start_time,
            total_duration=args.duration
        )

    # Plot spectrogram
    plt.figure(figsize=(12, 6))
    plt.specgram(total_signal, NFFT=2048, Fs=sr, noverlap=1024, cmap='magma')
    plt.title("Simulated Meteor Shower Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Intensity [dB]")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()

    print(f"Generated meteor shower spectrogram: {args.output}")

if __name__ == "__main__":
    main()

