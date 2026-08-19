#!/usr/bin/env python3
"""
G3PLX-style meteor trail simulator
----------------------------------

A reconstruction of the model described by G3ZJO in:
https://g3zjoradio.wordpress.com/2017/11/25/in-persuit-of-c-shape-meteor-reflections/

This is NOT the original G3PLX source code (which was lost). It implements the
physics described in the article:

  1. A meteor trail is represented by a line of point reflectors.
  2. TX and RX are placed to one side of the trail.
  3. Reflection is strongest where the trail is perpendicular to the
     bistatic look direction.
  4. The trail is bent with a sinusoid.
  5. A third spatial harmonic can be added to produce the W/epsilon shape.
  6. Doppler is calculated from the time derivative of the bistatic
     TX -> reflector -> RX path length.

Only numpy and matplotlib are required.

Run:
    python3 g3plx_sim.py --case line
    python3 g3plx_sim.py --case sine
    python3 g3plx_sim.py --case third

The script produces a spectrogram-like plot and, optionally, CSV data.
"""

import argparse
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Geometry:
    # Coordinates are metres.  The trail is centred at (0, 0).
    tx: np.ndarray
    rx: np.ndarray


@dataclass
class SimConfig:
    frequency_hz: float = 49.97e6
    audio_center_hz: float = 1500.0

    # Trail
    trail_length_m: float = 6000.0
    n_points: int = 801

    # Simulation time
    duration_s: float = 12.0
    dt_s: float = 0.02

    # Rotation of the trail through the bistatic reflecting orientation
    theta_start_deg: float = -8.0
    theta_end_deg: float = 8.0

    # Sinusoidal bending
    bend_amplitude_m: float = 1200.0
    bend_wavelength_m: float = 6000.0

    # Third harmonic relative amplitude
    third_harmonic: float = 0.32

    # Width of the specular/reflection response.
    # Smaller = thinner C/U traces.
    reflection_angle_width_deg: float = 1.8

    # Width of each simulated Doppler line in the rendered image
    doppler_width_hz: float = 0.8

    # Number of frequency bins in the rendered spectrogram
    n_freq: int = 900

    # Frequency display half-width around the audio carrier
    display_half_width_hz: float = 18.0


def rotation_matrix(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


def trail_shape(z, cfg, case):
    """
    Return x(z), y(z) in the trail's intrinsic coordinate system.

    z runs along the nominal meteor trail.  The bending is in x.
    """
    if case == "line":
        x = np.zeros_like(z)

    else:
        k = 2.0 * np.pi / cfg.bend_wavelength_m
        fundamental = cfg.bend_amplitude_m * np.sin(k * z)

        if case == "sine":
            x = fundamental

        elif case == "third":
            x = fundamental + (
                cfg.third_harmonic
                * cfg.bend_amplitude_m
                * np.sin(3.0 * k * z)
            )

        else:
            raise ValueError(f"Unknown case: {case}")

    return x, z


def trail_geometry(t, cfg, case):
    """
    Calculate positions and velocities of all point reflectors.

    Returns:
        pos : (N, 2)
        vel : (N, 2)
        tangent : (N, 2), unit tangent along the trail
    """
    z = np.linspace(
        -cfg.trail_length_m / 2,
        cfg.trail_length_m / 2,
        cfg.n_points
    )

    # Smoothly rotate the trail through the specular orientation.
    u = np.clip(t / cfg.duration_s, 0.0, 1.0)
    theta = np.deg2rad(
        cfg.theta_start_deg
        + u * (cfg.theta_end_deg - cfg.theta_start_deg)
    )

    # Intrinsic trail shape.
    x, y = trail_shape(z, cfg, case)
    p0 = np.vstack((x, y))

    # Rotation about trail centre.
    R = rotation_matrix(theta)
    pos = R @ p0

    # Velocity due to rotation.  For a 2-D rotation:
    # dR/dt * p = omega * (-y_rot, x_rot)
    omega = np.deg2rad(
        (cfg.theta_end_deg - cfg.theta_start_deg) / cfg.duration_s
    )
    vel = omega * np.vstack((-pos[1], pos[0]))

    # Numerical tangent after deformation + rotation.
    dp_dz = np.gradient(pos, z, axis=1)
    tangent = dp_dz / np.linalg.norm(dp_dz, axis=0, keepdims=True)

    return pos.T, vel.T, tangent.T


def bistatic_unit_vectors(points, station):
    """
    Unit vectors from a reflector toward a station.
    """
    d = station[None, :] - points
    r = np.linalg.norm(d, axis=1)
    return d / r[:, None], r


def reflector_response(points, tangent, geom, cfg):
    """
    Approximate bistatic/specular response.

    For a long plasma trail the strongest coherent reflection occurs when
    the trail tangent is approximately perpendicular to the bistatic
    look-direction.  We therefore use:

        q = tangent . normalize(u_TX + u_RX)

    and give maximum response around q=0.

    This is deliberately a smooth phenomenological response rather than a
    full electromagnetic scattering calculation.
    """
    u_tx, r_tx = bistatic_unit_vectors(points, geom.tx)
    u_rx, r_rx = bistatic_unit_vectors(points, geom.rx)

    bisector = u_tx + u_rx
    bisector /= np.linalg.norm(bisector, axis=1)[:, None]

    # |dot| = 0 is perfect broadside/specular orientation.
    q = np.abs(np.sum(tangent * bisector, axis=1))

    # Convert the angular error from radians.
    angle_error = np.arcsin(np.clip(q, 0.0, 1.0))
    sigma = np.deg2rad(cfg.reflection_angle_width_deg)

    weight = np.exp(-0.5 * (angle_error / sigma) ** 2)

    # Mild geometric spreading.
    weight *= 1.0 / np.sqrt(r_tx * r_rx)

    return weight, r_tx, r_rx


def bistatic_doppler(points, velocities, geom, cfg):
    """
    Doppler of each moving point reflector.

    dR/dt = u_TX . v + u_RX . v
    f_D = -(1/lambda) dR/dt

    The minus sign is the conventional radar Doppler sign.  We later add
    the result to the 1500-Hz audio centre frequency.
    """
    wavelength = 299792458.0 / cfg.frequency_hz

    u_tx, _ = bistatic_unit_vectors(points, geom.tx)
    u_rx, _ = bistatic_unit_vectors(points, geom.rx)

    range_rate = np.sum((u_tx + u_rx) * velocities, axis=1)
    return -range_rate / wavelength


def render_case(case, cfg, geom):
    """
    Generate a synthetic SpectrumLab-like intensity image.
    """
    times = np.arange(0.0, cfg.duration_s + cfg.dt_s / 2, cfg.dt_s)

    freq = np.linspace(
        cfg.audio_center_hz - cfg.display_half_width_hz,
        cfg.audio_center_hz + cfg.display_half_width_hz,
        cfg.n_freq
    )

    image = np.zeros((len(freq), len(times)))

    # Each reflector deposits a small Gaussian line into the spectrogram.
    for j, t in enumerate(times):
        points, velocities, tangent = trail_geometry(t, cfg, case)

        response, _, _ = reflector_response(
            points, tangent, geom, cfg
        )

        fd = bistatic_doppler(points, velocities, geom, cfg)
        f = cfg.audio_center_hz + fd

        # Ignore reflectors that cannot contribute.
        active = response > response.max() * 1e-4

        if not np.any(active):
            continue

        ff = f[active]
        ww = response[active]

        # Vectorised rendering of point-reflector lines.
        d = freq[:, None] - ff[None, :]
        image[:, j] = np.sum(
            ww[None, :] * np.exp(
                -0.5 * (d / cfg.doppler_width_hz) ** 2
            ),
            axis=1
        )

    # Log compression gives a display closer to a radio spectrogram.
    image /= image.max() + 1e-30
    image_db = 10.0 * np.log10(image + 1e-6)

    return times, freq, image_db


def save_csv(filename, times, freq, image_db):
    """
    Save the rendered image as a simple CSV:
      first column = frequency
      remaining columns = time samples
    """
    data = np.column_stack((freq, image_db))
    header = "frequency_Hz," + ",".join(f"{t:.6f}" for t in times)
    np.savetxt(filename, data, delimiter=",", header=header, comments="")


def main():
    parser = argparse.ArgumentParser(
        description="G3PLX-style meteor Doppler simulator"
    )
    parser.add_argument(
        "--case",
        choices=("line", "sine", "third", "all"),
        default="all"
    )
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    cfg = SimConfig()

    # IMPORTANT:
    # These are deliberately simple normalized coordinates, not the actual
    # Dourbes/Humain geometry.  The stations are placed to the right of the
    # trail so that a vertical trail is the reflecting orientation.
    geom = Geometry(
        tx=np.array([10000.0, 3500.0]),
        rx=np.array([14000.0, -1500.0]),
    )

    cases = ("line", "sine", "third") if args.case == "all" else (args.case,)

    for case in cases:
        times, freq, image = render_case(case, cfg, geom)

        plt.figure(figsize=(11, 6))
        extent = [
            times[0],
            times[-1],
            freq[0],
            freq[-1],
        ]

        plt.imshow(
            image,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="bilinear",
            cmap="viridis",
            vmin=-45,
            vmax=0,
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Audio frequency (Hz)")
        plt.title(f"G3PLX-style meteor simulation: {case}")
        plt.colorbar(label="Relative power (dB)")
        plt.tight_layout()

        output = f"g3plx_{case}.png"
        plt.savefig(output, dpi=180)
        print(f"Saved {output}")

        if args.csv:
            csv_name = f"g3plx_{case}.csv"
            save_csv(csv_name, times, freq, image)
            print(f"Saved {csv_name}")

    plt.show()


if __name__ == "__main__":
    main()
