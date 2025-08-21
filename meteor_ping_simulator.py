#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import wave, struct
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class SimConfig:
    fs: int = 12000
    dur_s: float = 20.0
    f0: float = 1000.0
    snr_db: float = -5.0
    event_rate_hz: float = 0.5
    p_overdense: float = 0.35
    tau_underdense_s: Tuple[float, float] = (0.03, 0.12)
    overdense_dur_s: Tuple[float, float] = (0.2, 1.2)
    fd_hz: Tuple[float, float] = (-150.0, 150.0)
    fd_slope_hz_s: Tuple[float, float] = (-300.0, 300.0)
    fd_curve_hz_s2: Tuple[float, float] = (-400.0, 400.0)
    nfft: int = 1024
    noverlap: int = 768

def poisson_event_times(rate_hz: float, T: float, rng: np.random.Generator):
    times = []
    t = 0.0
    while True:
        dt = rng.exponential(1.0 / max(rate_hz, 1e-9))
        t += dt
        if t >= T:
            break
        times.append(t)
    return times

def underdense_envelope(t, tau):
    env = np.exp(-np.clip(t, 0, None) / max(tau, 1e-6))
    env[t < 0] = 0.0
    return env

def raised_cosine_window(N, frac=0.05):
    N = max(N, 1)
    ramp = max(int(frac * N), 1)
    w = np.ones(N, dtype=float)
    for i in range(ramp):
        w[i] = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / ramp)
        w[-(i + 1)] = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / ramp)
    return w

def overdense_envelope(t, dur):
    env = np.zeros_like(t, dtype=float)
    idx = np.where((t >= 0) & (t <= dur))[0]
    if len(idx) == 0:
        return env
    env[idx] = raised_cosine_window(len(idx), 0.05)
    return env

def doppler_phase(t, fd0, fd1, fd2):
    return 2 * np.pi * (fd0 * t + 0.5 * fd1 * t**2 + (1.0/6.0) * fd2 * t**3)

def synth_event(cfg, t, t0, rng):
    te = t - t0
    if rng.random() < cfg.p_overdense:
        dur = rng.uniform(*cfg.overdense_dur_s)
        env = overdense_envelope(te, dur)
    else:
        tau = rng.uniform(*cfg.tau_underdense_s)
        env = underdense_envelope(te, tau)
    fd0 = rng.uniform(*cfg.fd_hz)
    fd1 = rng.uniform(*cfg.fd_slope_hz_s)
    fd2 = rng.uniform(*cfg.fd_curve_hz_s2)
    phi = doppler_phase(np.clip(te, 0, None), fd0, fd1, fd2)
    carrier = np.cos(2*np.pi*cfg.f0*t + phi)
    A = 10**(rng.uniform(-10, 0)/20.0)
    return A*env*carrier

def add_awgn(x, snr_db, rng):
    sig_pwr = np.mean(x**2) + 1e-12
    snr_lin = 10**(snr_db/10.0)
    noise_pwr = sig_pwr / max(snr_lin, 1e-9)
    n = rng.normal(0.0, np.sqrt(noise_pwr), size=x.shape)
    return x + n

def write_wav_int16(path, fs, x):
    x = np.asarray(x, dtype=float)
    maxv = np.max(np.abs(x)) + 1e-12
    y = np.clip(x / maxv, -1.0, 1.0)
    y = (y * 32767.0).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(y.tobytes())

def main():
    rng = np.random.default_rng(42)
    cfg = SimConfig()
    N = int(cfg.fs * cfg.dur_s)
    t = np.arange(N) / cfg.fs
    starts = poisson_event_times(cfg.event_rate_hz, cfg.dur_s, rng)
    x = np.zeros(N, dtype=float)
    for t0 in starts:
        x += synth_event(cfg, t, t0, rng)
    x = add_awgn(x, cfg.snr_db, rng)
    np.asarray(x, dtype=np.float32).tofile('meteor_sim.dat')
    write_wav_int16('meteor_sim.wav', cfg.fs, x)
    plt.figure(figsize=(10, 5))
    plt.specgram(x, NFFT=cfg.nfft, Fs=cfg.fs, noverlap=cfg.noverlap)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Simulated Meteor Echoes (underdense & overdense, doppler bending)')
    plt.tight_layout()
    plt.savefig('meteor_sim.png', dpi=150)

if __name__ == '__main__':
    main()
