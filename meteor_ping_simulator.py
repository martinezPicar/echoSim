#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
G3PLX Meteor Ping Simulator — Python 3 Implementation
================================================================================

Based on the original DOS software by Peter Martínez (G3PLX), 1980s.

This simulator models meteor scatter radio reflections using a single-line
point-reflector method. It produces characteristic C/U and W/epsilon shapes
when viewed on a spectrogram/waterfall display (e.g., SpectrumLab, Audacity).

The output is a 1500 Hz carrier tone that is Doppler-modulated according to
the geometry of a simulated meteor trail, including wind shear effects.

Usage:
    python meteor_ping_simulator.py --mode c -o output.wav
    python meteor_ping_simulator.py --mode epsilon --plot
    python meteor_ping_simulator.py --duration 5 --doppler 100 --tilt 15
================================================================================
"""

import numpy as np
import wave
import argparse


class MeteorPingSimulator:
    """
    Simulates meteor scatter radio echoes.

    Physics model:
    1. Meteor trail = line of ionized point reflectors at ~100km altitude
    2. Trail rotates from initial tilt through vertical
    3. Wind shear bends trail sinusoidally → C/U shape on spectrogram
    4. 3rd harmonic shear → W/epsilon shape
    5. Specular reflection determines which part of trail is "visible"
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        carrier_freq: float = 1500.0,
        duration: float = 3.0,
        max_doppler: float = 75.0,
        tilt_deg: float = 10.0,
        epsilon: bool = False,
        epsilon_amp: float = 0.35,
        decay: float = 0.6,
        snr_db: float = 25.0,
        n_reflectors: int = 150,
    ):
        self.sample_rate = sample_rate
        self.carrier_freq = carrier_freq
        self.duration = duration
        self.max_doppler = max_doppler
        self.tilt_deg = tilt_deg
        self.epsilon = epsilon
        self.epsilon_amp = epsilon_amp
        self.decay = decay
        self.snr_db = snr_db
        self.n_reflectors = n_reflectors

    def generate(self) -> np.ndarray:
        """Generate the meteor ping audio signal."""
        num_samples = int(self.sample_rate * self.duration)
        t = np.linspace(0, self.duration, num_samples)
        s = np.linspace(-1, 1, self.n_reflectors)
        signal = np.zeros(num_samples, dtype=np.float64)

        # Trail rotation over time: tilted → vertical → tilted other way
        rotation = np.radians(self.tilt_deg) * np.cos(np.pi * t / self.duration)

        # Wind shear profile along trail
        shear = np.sin(np.pi * s)
        if self.epsilon:
            shear += self.epsilon_amp * np.sin(3 * np.pi * s)

        for i, ti in enumerate(t):
            rot = rotation[i]

            # Specular reflection point moves as trail rotates
            specular = -np.sin(rot) * 0.5
            reflection = np.exp(-((s - specular) / 0.15) ** 2)

            # Trail ionization decays over time
            trail_decay = np.exp(-ti / self.decay)
            amp = reflection * trail_decay

            # Doppler shift for each reflector
            doppler = self.max_doppler * shear * np.sin(rot * 2)

            total_amp = np.sum(amp)
            if total_amp > 0:
                avg_doppler = np.sum(amp * doppler) / total_amp
                freq = self.carrier_freq + avg_doppler
                envelope = total_amp / self.n_reflectors * 5
                signal[i] = envelope * np.sin(2 * np.pi * freq * ti)

        # Normalize
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val

        # Add noise
        if self.snr_db > 0:
            sig_power = np.mean(signal ** 2)
            noise_power = sig_power / (10 ** (self.snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), num_samples)
            signal = signal + noise
            signal = signal / np.max(np.abs(signal)) * 0.95

        return signal.astype(np.float32)

    def save_wav(self, signal: np.ndarray, filename: str):
        """Save signal as 16-bit mono WAV."""
        sig_int16 = (signal * 32767).astype(np.int16)
        with wave.open(filename, "w") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(sig_int16.tobytes())
        print(f"Saved WAV: {filename}")

    def spectrogram(self, signal: np.ndarray, nfft: int = 1024):
        """Compute spectrogram data. Returns (freqs, times, power_db)."""
        from scipy import signal as scipy_signal
        freqs, times, Sxx = scipy_signal.spectrogram(
            signal, fs=self.sample_rate, nperseg=nfft, noverlap=nfft // 2, scaling="spectrum"
        )
        return freqs, times, 10 * np.log10(Sxx + 1e-10)


def plot_spectrogram(freqs, times, spec, carrier, title, filename):
    """Plot and save a spectrogram image."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(
        times, freqs, spec, shading="gouraud", cmap="hot",
        vmin=np.max(spec) - 40, vmax=np.max(spec)
    )
    ax.set_ylim(carrier - 250, carrier + 250)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)
    plt.colorbar(mesh, ax=ax, label="Intensity [dB]")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="G3PLX Meteor Ping Simulator — Python 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python meteor_ping_simulator.py --mode c -o ping.wav
  python meteor_ping_simulator.py --mode epsilon --plot
  python meteor_ping_simulator.py --duration 5 --doppler 100 --tilt 15 --epsilon
  python meteor_ping_simulator.py --mode underdense --duration 0.3 --doppler 10
        """,
    )
    parser.add_argument(
        "--mode", choices=["c", "epsilon", "underdense", "custom"],
        default="custom", help="Preset mode (default: custom)"
    )
    parser.add_argument("-o", "--output", default="meteor_ping.wav", help="Output WAV file")
    parser.add_argument("--duration", type=float, default=3.0, help="Event duration [s]")
    parser.add_argument("--carrier", type=float, default=1500.0, help="Carrier frequency [Hz]")
    parser.add_argument("--doppler", type=float, default=75.0, help="Max Doppler shift [Hz]")
    parser.add_argument("--tilt", type=float, default=10.0, help="Initial trail tilt [deg]")
    parser.add_argument("--epsilon", action="store_true", help="Enable W/epsilon shape")
    parser.add_argument("--epsilon-amp", type=float, default=0.35, help="3rd harmonic amplitude")
    parser.add_argument("--decay", type=float, default=0.6, help="Trail decay time constant [s]")
    parser.add_argument("--snr", type=float, default=25.0, help="SNR [dB]")
    parser.add_argument("--sr", type=int, default=48000, help="Sample rate [Hz]")
    parser.add_argument("--plot", action="store_true", help="Generate spectrogram plot")

    args = parser.parse_args()

    # Apply presets
    if args.mode == "c":
        duration, doppler, tilt, eps, eps_amp, decay = 3.0, 75.0, 12.0, False, 0.35, 0.8
        title = "C-Shape Meteor Ping"
    elif args.mode == "epsilon":
        duration, doppler, tilt, eps, eps_amp, decay = 4.0, 100.0, 15.0, True, 0.4, 1.0
        title = "Epsilon/W-Shape Meteor Ping"
    elif args.mode == "underdense":
        duration, doppler, tilt, eps, eps_amp, decay = 0.3, 10.0, 2.0, False, 0.35, 0.15
        title = "Underdense Meteor Ping"
    else:
        duration = args.duration
        doppler = args.doppler
        tilt = args.tilt
        eps = args.epsilon
        eps_amp = args.epsilon_amp
        decay = args.decay
        title = "Custom Meteor Ping"

    print("=" * 60)
    print("G3PLX Meteor Ping Simulator — Python 3")
    print("=" * 60)
    print(f"Mode:        {args.mode}")
    print(f"Duration:    {duration} s")
    print(f"Carrier:     {args.carrier} Hz")
    print(f"Max Doppler: {doppler} Hz")
    print(f"Tilt:        {tilt}°")
    print(f"Epsilon:     {eps}")
    print(f"Decay:       {decay} s")
    print(f"SNR:         {args.snr} dB")
    print("=" * 60)

    sim = MeteorPingSimulator(
        sample_rate=args.sr,
        carrier_freq=args.carrier,
        duration=duration,
        max_doppler=doppler,
        tilt_deg=tilt,
        epsilon=eps,
        epsilon_amp=eps_amp,
        decay=decay,
        snr_db=args.snr,
    )

    signal = sim.generate()
    sim.save_wav(signal, args.output)

    if args.plot:
        try:
            freqs, times, spec = sim.spectrogram(signal)
            plot_file = args.output.replace(".wav", ".png")
            plot_spectrogram(freqs, times, spec, args.carrier, title, plot_file)
        except ImportError:
            print("matplotlib/scipy not available, skipping plot")

    print("\nDone! Play with:")
    print(f"  aplay {args.output}       # Linux")
    print(f"  afplay {args.output}      # macOS")
    print(f"  ffplay {args.output}      # cross-platform (ffmpeg)")


if __name__ == "__main__":
    main()