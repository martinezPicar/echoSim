#!/usr/bin/env python3
"""
meteor_ping_simulator.py

A Python 3 re-implementation of the physical model behind Peter Martinez
G3PLX's "Meteor Ping Simulator" (originally written on DOS in the 1980s,
since lost -- see G3ZJO's write-up "In Pursuit of C shape Meteor
reflections", g3zjoradio.wordpress.com, Nov/Dec 2017).

G3PLX described his approach to G3ZJO as a "single line point reflector
method" to model meteor trails:

  * the trail is treated as a line of point reflectors
  * the whole line rotates -- when it swings through the orientation that
    is perpendicular to the bisector of the transmitter-scatterer-receiver
    angle ("vertical" in his display convention) the forward-scatter link
    is briefly closed and a "ping" is heard
  * a wind-shear layer bends the trail: the sinusoidal bending pushes the
    top of the trail one way in Doppler and the bottom the other way,
    while the centre stays put -- producing a "C" (or "U") shape on a
    waterfall/spectrogram
  * adding a 3rd-harmonic sinusoidal term to the bending produces a "W"
    (epsilon) shape

This module reproduces that physics directly rather than hand-drawing the
curves: it places many point scatterers along a trail, moves them
according to a rotation + spatial-harmonic bending law, computes the real
bistatic (TX -> scatterer -> RX) path length for every scatterer at every
time step, and coherently sums their contributions as phasors at the
actual radar wavelength. The characteristic C / epsilon Doppler traces
emerge from that interference sum, the same way they would from a real
forward-scatter echo, rather than being drawn on top of a spectrogram.

Author: (rebuilt for Tony, SIDC / Royal Observatory of Belgium)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import wave
import struct
from pathlib import Path
from typing import Optional

import numpy as np


C_LIGHT = 299_792.458  # km/s


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def _flat_earth_xyz(lat_deg: float, lon_deg: float, alt_km: float,
                     ref_lat_deg: float, ref_lon_deg: float) -> np.ndarray:
    """
    Very small-area flat-earth approximation converting (lat, lon, alt) to
    a local ENU-like (x=east, y=north, z=up) frame in km, relative to a
    reference point. Good enough over the few-hundred-km baselines typical
    of a forward-scatter meteor link (e.g. BRAMS); not meant for anything
    requiring geodetic precision.
    """
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.radians(ref_lat_deg))
    x = (lon_deg - ref_lon_deg) * km_per_deg_lon
    y = (lat_deg - ref_lat_deg) * km_per_deg_lat
    z = alt_km
    return np.array([x, y, z])


@dataclasses.dataclass
class Station:
    """A transmitter or receiver, given either as lat/lon/alt or local x,y,z (km)."""
    name: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_km: float = 0.0
    xyz_km: Optional[np.ndarray] = None

    def resolve(self, ref_lat: float, ref_lon: float) -> np.ndarray:
        if self.xyz_km is not None:
            return np.asarray(self.xyz_km, dtype=float)
        if self.lat is None or self.lon is None:
            raise ValueError(f"Station {self.name!r} needs either xyz_km or lat/lon")
        return _flat_earth_xyz(self.lat, self.lon, self.alt_km, ref_lat, ref_lon)


# --------------------------------------------------------------------------- #
# The simulator
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class MeteorPingParams:
    # Radar / link parameters
    freq_hz: float = 49_970_000.0        # BRAMS beacon frequency by default
    duration_s: float = 2.5
    sample_rate_hz: int = 8000
    audio_tone_hz: float = 1500.0        # reference tone, as used by G3PLX/SpectrumLab

    # Trail geometry (local km coordinates, trail centred at origin unless
    # a real TX/RX geometry is supplied)
    trail_length_km: float = 8.0
    n_points: int = 121                  # odd, so there's a centre point
    trail_altitude_km: float = 100.0

    # Bistatic geometry: perpendicular distance from trail to the
    # TX-RX baseline's midpoint, and TX/RX offsets along that baseline.
    # These defaults describe a modest forward-scatter geometry loosely
    # in the spirit of a BRAMS-type link (beacon + remote receiver).
    tx_offset_km: float = -150.0
    rx_offset_km: float = 150.0
    link_range_km: float = 300.0         # ground-range "depth" to the trail

    # Rotation (brings the trail through the specular / "vertical" angle)
    theta0_deg: float = -8.0             # initial tilt away from specular
    omega_rot_deg_s: float = 6.0         # rotation rate

    # Wind-shear bending: spatial harmonic(s) along the trail, growing in
    # amplitude with time. harmonic 1 -> "C"/"U" shape; add harmonic 3 for
    # the "epsilon"/"W" shape.
    shear1_amp_deg: float = 0.0          # peak bending angle, 1st harmonic
    shear1_rate_deg_s: float = 0.0       # growth rate of that amplitude
    shear3_amp_deg: float = 0.0          # peak bending angle, 3rd harmonic
    shear3_rate_deg_s: float = 0.0
    shear3_phase_deg: float = 0.0

    # Trail illumination taper (Gaussian window along the trail so the
    # ends don't contribute as strongly as the middle -- ionisation density
    # is rarely a hard-edged rectangle)
    illumination_sigma_frac: float = 0.35  # fraction of half-length

    seed: Optional[int] = None


class MeteorPingSimulator:
    """
    Rebuilds the G3PLX "single line point reflector" meteor ping model.

    Usage:
        sim = MeteorPingSimulator(MeteorPingParams(...))
        t, audio = sim.run()
        sim.save_wav("ping.wav", audio)
        sim.save_spectrogram("ping.png", audio)
    """

    def __init__(self, params: MeteorPingParams):
        self.p = params
        self._rng = np.random.default_rng(params.seed)

        # positions of TX / RX / trail-centre in a local flat km frame:
        # baseline runs along x, trail sits "above" the midpoint, offset in
        # ground range (y) and altitude (z).
        p = params
        self.tx_xyz = np.array([p.tx_offset_km, 0.0, 0.0])
        self.rx_xyz = np.array([p.rx_offset_km, 0.0, 0.0])
        self.trail_centre_xyz = np.array([0.0, p.link_range_km, p.trail_altitude_km])

        self.wavelength_km = C_LIGHT / p.freq_hz

        # scatterer positions along the trail axis, s in [-L/2, +L/2] km
        self.s = np.linspace(-p.trail_length_km / 2, p.trail_length_km / 2, p.n_points)

        # illumination taper (relative scatterer amplitude)
        half_len = p.trail_length_km / 2
        sigma = max(p.illumination_sigma_frac, 1e-3) * half_len
        self.illumination = np.exp(-0.5 * (self.s / sigma) ** 2)

    # ------------------------------------------------------------------ #
    def trail_tilt_deg(self, t: np.ndarray) -> np.ndarray:
        """
        Bulk rotation angle of the trail (deg) at time t (seconds), shared
        by every scatterer -- this is what sweeps the trail through the
        specular ("vertical") orientation and produces the basic ping.
        """
        return self.p.theta0_deg + self.p.omega_rot_deg_s * t

    def bending_deg(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Extra, position-dependent tilt (deg) from wind-shear bending.
        s: (n_points,) trail-axis positions in km
        t: (n_time,) times in seconds
        returns array of shape (n_points, n_time)
        """
        p = self.p
        half_len = p.trail_length_km / 2
        s_norm = s / half_len  # -1 .. +1

        S, T = np.meshgrid(s_norm, t, indexing="ij")

        # bending amplitude grows linearly in time from its starting value
        amp1 = p.shear1_amp_deg + p.shear1_rate_deg_s * T
        amp3 = p.shear3_amp_deg + p.shear3_rate_deg_s * T

        bend = amp1 * np.sin(np.pi * S) + amp3 * np.sin(
            3 * np.pi * S + np.radians(p.shear3_phase_deg)
        )
        return bend

    # ------------------------------------------------------------------ #
    def scatterer_positions(self, t: np.ndarray) -> np.ndarray:
        """
        Returns scatterer xyz positions, shape (n_points, n_time, 3), in km,
        in the local flat frame (x = along baseline, y = ground range,
        z = altitude). The trail is assumed to lie in the x-z plane
        (i.e. its long axis tilts between "horizontal-ish" and "vertical"
        in that plane, matching the rotate-through-specular picture);
        y (ground range to the trail) stays fixed at the trail centre.
        """
        tilt_bulk = self.trail_tilt_deg(t)                      # (n_time,)
        tilt_bend = self.bending_deg(self.s, t)                 # (n_pts, n_time)
        tilt_total_deg = tilt_bulk[None, :] + tilt_bend         # (n_pts, n_time)
        tilt_rad = np.radians(tilt_total_deg)

        S = self.s[:, None]  # (n_pts, 1) km along trail axis at t=0 orientation

        # position relative to trail centre, rotated in the x-z plane
        dx = S * np.sin(tilt_rad)
        dz = S * np.cos(tilt_rad)
        dy = np.zeros_like(dx)

        pos = self.trail_centre_xyz[None, None, :] + np.stack([dx, dy, dz], axis=-1)
        return pos  # (n_pts, n_time, 3)

    # ------------------------------------------------------------------ #
    def path_length_km(self, pos: np.ndarray) -> np.ndarray:
        """Bistatic path length TX -> scatterer -> RX, shape (n_points, n_time)."""
        d_tx = np.linalg.norm(pos - self.tx_xyz[None, None, :], axis=-1)
        d_rx = np.linalg.norm(pos - self.rx_xyz[None, None, :], axis=-1)
        return d_tx + d_rx

    # ------------------------------------------------------------------ #
    def run(self):
        """
        Runs the simulation and returns (t, audio) where t is the time
        vector (s) and audio is a real-valued float array in [-1, 1]
        suitable for writing to a WAV file.
        """
        p = self.p
        n_samples = int(round(p.duration_s * p.sample_rate_hz))
        t = np.arange(n_samples) / p.sample_rate_hz

        pos = self.scatterer_positions(t)          # (n_pts, n_time, 3)
        L = self.path_length_km(pos)                # (n_pts, n_time) km

        # Subtract a single scalar reference path length (not a per-scatterer
        # one!) just to keep the numbers small before the 2*pi/lambda scaling
        # -- this must NOT remove the *relative* phase differences between
        # scatterers, or every point becomes artificially in-phase at t=0
        # regardless of geometry, which would fake a specular peak at t=0.
        L_ref = L[self.s.size // 2, 0]  # path length of the centre scatterer at t=0
        dL = L - L_ref
        phase = 2.0 * np.pi * dL / self.wavelength_km   # (n_pts, n_time)

        weights = self.illumination[:, None]
        baseband = np.sum(weights * np.exp(1j * phase), axis=0)  # (n_time,)

        # normalise
        peak = np.max(np.abs(baseband))
        if peak > 0:
            baseband = baseband / peak

        # mix up to an audible reference tone (as G3PLX's simulator did,
        # feeding SpectrumLab/SB-Spectrum at ~1500 Hz)
        carrier = np.exp(1j * 2.0 * np.pi * p.audio_tone_hz * t)
        audio_complex = baseband * carrier
        audio = np.real(audio_complex)

        # gentle fade in/out to avoid click artefacts at the ping edges
        fade_n = max(1, int(0.01 * p.sample_rate_hz))
        window = np.ones(n_samples)
        window[:fade_n] = np.linspace(0, 1, fade_n)
        window[-fade_n:] = np.linspace(1, 0, fade_n)
        audio = audio * window

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95

        return t, audio.astype(np.float64)

    # ------------------------------------------------------------------ #
    @staticmethod
    def save_wav(path: str, audio: np.ndarray, sample_rate_hz: int):
        path = str(path)
        n_samples = len(audio)
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate_hz)
            wf.writeframes(struct.pack(f"<{n_samples}h", *pcm.tolist()))

    # ------------------------------------------------------------------ #
    def save_spectrogram(self, path: str, audio: np.ndarray, title: str = ""):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.signal import spectrogram

        fs = self.p.sample_rate_hz
        nperseg = min(512, len(audio))
        noverlap = int(nperseg * 0.9)
        f, tt, Sxx = spectrogram(
            audio, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann"
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        fig, ax = plt.subplots(figsize=(7, 5))
        f_lo = max(0, self.p.audio_tone_hz - 300)
        f_hi = self.p.audio_tone_hz + 300
        mask = (f >= f_lo) & (f <= f_hi)
        im = ax.pcolormesh(tt, f[mask], Sxx_db[mask, :], shading="gouraud", cmap="magma")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_title(title or "Meteor ping simulator (G3PLX-style)")
        fig.colorbar(im, ax=ax, label="dB")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Presets matching the three scenarios G3PLX demonstrated to G3ZJO
# --------------------------------------------------------------------------- #

PRESETS = {
    "simple_ping": dict(
        # trail rotates through the specular angle once, no shear:
        # produces the basic single "U"/"C" ping (u_sim1.jpg / sim_vert.jpg)
        theta0_deg=-8.0, omega_rot_deg_s=6.0,
        shear1_amp_deg=0.0, shear1_rate_deg_s=0.0,
        shear3_amp_deg=0.0, shear3_rate_deg_s=0.0,
    ),
    "c_shape": dict(
        # add 1st-harmonic wind-shear bending, growing with time
        theta0_deg=-2.0, omega_rot_deg_s=1.0,
        shear1_amp_deg=0.0, shear1_rate_deg_s=9.0,
        shear3_amp_deg=0.0, shear3_rate_deg_s=0.0,
    ),
    "epsilon_shape": dict(
        # add a 3rd-harmonic term on top -> W / epsilon shape
        theta0_deg=-2.0, omega_rot_deg_s=1.0,
        shear1_amp_deg=0.0, shear1_rate_deg_s=9.0,
        shear3_amp_deg=0.0, shear3_rate_deg_s=6.0, shear3_phase_deg=90.0,
    ),
}


def build_params(preset: str, **overrides) -> MeteorPingParams:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {list(PRESETS)}")
    base = dataclasses.asdict(MeteorPingParams())
    base.update(PRESETS[preset])
    base.update(overrides)
    return MeteorPingParams(**base)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Meteor Ping Simulator (G3PLX-style single-line point-reflector model)"
    )
    ap.add_argument("--preset", choices=list(PRESETS), default="c_shape")
    ap.add_argument("--duration", type=float, default=2.5, help="duration in seconds")
    ap.add_argument("--freq-hz", type=float, default=49_970_000.0,
                     help="radar/beacon carrier frequency, Hz (default: BRAMS 49.97 MHz)")
    ap.add_argument("--sample-rate", type=int, default=8000)
    ap.add_argument("--audio-tone-hz", type=float, default=1500.0)
    ap.add_argument("--out-prefix", type=str, default=None,
                     help="output file prefix (defaults to the preset name)")
    ap.add_argument("--params-json", type=str, default=None,
                     help="path to a JSON file of MeteorPingParams overrides")
    args = ap.parse_args()

    overrides = dict(
        duration_s=args.duration,
        freq_hz=args.freq_hz,
        sample_rate_hz=args.sample_rate,
        audio_tone_hz=args.audio_tone_hz,
    )
    if args.params_json:
        with open(args.params_json) as fh:
            overrides.update(json.load(fh))

    params = build_params(args.preset, **overrides)
    sim = MeteorPingSimulator(params)
    t, audio = sim.run()

    prefix = args.out_prefix or args.preset
    wav_path = f"{prefix}.wav"
    png_path = f"{prefix}.png"
    sim.save_wav(wav_path, audio, params.sample_rate_hz)
    sim.save_spectrogram(png_path, audio, title=f"Meteor ping ({args.preset})")

    print(f"Wrote {wav_path} and {png_path}")


if __name__ == "__main__":
    main()
