import numpy as np
import scipy.io.wavfile as wav
from scipy.optimize import minimize_scalar
import tkinter as tk
from tkinter import filedialog

import matplotlib
matplotlib.use('TkAgg')  # GUI backend required for interactive controls
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False



# ---------------------------------------------------------------------------
# BRAMS forward-scatter geometry
# ---------------------------------------------------------------------------
# Coordinates are WGS84 geodetic coordinates.  Dourbes is the BRAMS beacon
# and Humain is used as the default receiver.
DOURBES_LAT_DEG = 50.0972
DOURBES_LON_DEG = 4.5847
DOURBES_ALT_M = 220.0

HUMAIN_LAT_DEG = 50.1639
HUMAIN_LON_DEG = 5.2181
HUMAIN_ALT_M = 293.0

C_LIGHT = 299_792_458.0
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    """WGS84 geodetic -> ECEF Cartesian coordinates (m)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)

    x = (n + h_m) * cos_lat * np.cos(lon)
    y = (n + h_m) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def enu_basis(lat_deg, lon_deg):
    """Return local East, North, Up unit vectors at a reference location."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
         np.cos(lat)
    ])
    up = np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ])
    return east, north, up


def build_forward_scatter_geometry(
    z_km, x_km, frequency_hz,
    meteor_azimuth_deg=45.0,
    meteor_elevation_deg=30.0,
    specular_altitude_km=90.0,
    tx_lat_deg=DOURBES_LAT_DEG, tx_lon_deg=DOURBES_LON_DEG,
    tx_alt_m=DOURBES_ALT_M,
    rx_lat_deg=HUMAIN_LAT_DEG, rx_lon_deg=HUMAIN_LON_DEG,
    rx_alt_m=HUMAIN_ALT_M,
    meteor_lat_deg=None, meteor_lon_deg=None,
    meteor_base_alt_m=90_000.0,
    trail_azimuth_deg=None,
):
    """
    Build the actual bistatic geometry for each trail element.

    The meteor trail is represented in a local ENU frame.  By default its
    horizontal x-axis is the horizontal Tx->Rx direction, so the model remains
    compatible with the original 2-D trail representation.

    The absolute trail centre is placed at the geographic midpoint between
    Tx and Rx unless meteor_lat/lon are supplied.  The specular point is then
    identified by minimizing R_T + R_R along the model trail.

    Returns:
        positions_ecef, tx_to_p, rx_to_p, beta, bisector, wavelength
    """
    tx = geodetic_to_ecef(tx_lat_deg, tx_lon_deg, tx_alt_m)
    rx = geodetic_to_ecef(rx_lat_deg, rx_lon_deg, rx_alt_m)

    # Reference point: geographic midpoint of Tx/Rx at the requested
    # specular altitude.
    ref_lat = 0.5 * (tx_lat_deg + rx_lat_deg)
    ref_lon = 0.5 * (tx_lon_deg + rx_lon_deg)
    ref = geodetic_to_ecef(ref_lat, ref_lon, specular_altitude_km * 1000.0)
    east, north, up = enu_basis(ref_lat, ref_lon)

    # Explicit meteor azimuth: clockwise from North.
    az = np.radians(meteor_azimuth_deg)
    el = np.radians(meteor_elevation_deg)

    xhat = np.sin(az) * east + np.cos(az) * north
    xhat /= np.linalg.norm(xhat)

    # Full meteor trajectory direction.
    trajectory_dir = (
        np.cos(el) * xhat + np.sin(el) * up
    )
    trajectory_dir /= np.linalg.norm(trajectory_dir)

    # Complete the horizontal basis.
    yhat = np.cross(up, xhat)
    yhat /= np.linalg.norm(yhat)

    positions = (
        ref[None, :]
        + (x_km * 1000.0)[:, None] * xhat[None, :]
        + np.zeros((len(z_km), 1))
        + (z_km * 1000.0)[:, None] * up[None, :]
    )

    # Unit vectors in the direction of increasing propagation distance.
    tx_to_p = positions - tx
    rx_to_p = positions - rx

    r_t = np.linalg.norm(tx_to_p, axis=1)
    r_r = np.linalg.norm(rx_to_p, axis=1)

    u_t = tx_to_p / r_t[:, None]
    u_r = rx_to_p / r_r[:, None]

    # Forward-scatter angle: angle between the two outgoing directions
    # from Tx/Rx to the scattering point.
    cos_beta = np.clip(np.sum(u_t * u_r, axis=1), -1.0, 1.0)
    beta = np.arccos(cos_beta)

    # Bistatic Doppler vector is the gradient of total path length:
    # grad(R_T + R_R) = u_t + u_r.
    bisector_vector = u_t + u_r
    bisector_norm = np.linalg.norm(bisector_vector, axis=1)
    bisector = bisector_vector / bisector_norm[:, None]

    wavelength = C_LIGHT / frequency_hz

    # Identify the geometrical specular point: minimum total propagation path.
    total_path = r_t + r_r
    specular_index = np.argmin(total_path)

    return (
        positions, tx_to_p, rx_to_p, beta, bisector,
        wavelength, specular_index, total_path,
        trajectory_dir, ref, xhat, up
    )


def synthesize_meteor_signal(
    skew_deg, wind_speed, ratio_fund, ratio_3rd, ratio_5th, alpha_ping,
    meteor_azimuth_deg=45.0,
    meteor_elevation_deg=30.0,
    specular_altitude_km=90.0,
    frequency_hz=49.97e6, sample_rate=44100, duration=15.0,
    carrier_freq=1000.0, snr_db=18.0
):
    num_samples = int(sample_rate * duration)
    dt = 1.0 / sample_rate
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Radio wavelength. Doppler is calculated below from the actual
    # transmitter/meteor/receiver geometry.
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    wavelength_m = C_LIGHT / frequency_hz

    # 1. Event Timing & Delayed Diffusion Envelope
    t_entry = 2.0  # Impact at t = 2.0s
    active_mask = t >= t_entry
    t_active = np.maximum(0.0, t - t_entry)

    rise_env = 1.0 - np.exp(-t_active / 0.005) # Fast initial ionization onset
    t_diffusion_start = 3.0                    # Signal holds for ~3s before fading
    diffusion_time = np.maximum(0.0, t_active - t_diffusion_start)
    decay_env = np.exp(-diffusion_time / 1.0)  # Ambipolar diffusion decay
    
    global_envelope = np.zeros(num_samples)
    global_envelope[active_mask] = rise_env[active_mask] * decay_env[active_mask]

    # 2. Altitude Layer Definition & Spatial Profile
    nz = 4000
    z = np.linspace(-1.2, 1.2, nz)
    dz = z[1] - z[0]

    # Initial spatial skew slope
    skew_rad = np.radians(skew_deg)
    initial_tilt_slope = np.tan(skew_rad)
    x0 = initial_tilt_slope * z
    dx0_dz = np.gradient(x0, dz)

    # Build the real BRAMS Dourbes -> Humain bistatic geometry.
    # z is centred on a 90-km trail altitude and x is the horizontal
    # coordinate in the local Tx-Rx vertical plane.
    (
        trail_ecef, tx_to_p, rx_to_p, beta, bistatic_bisector,
        wavelength_m, specular_index, total_path,
        trajectory_dir, trajectory_ref, trajectory_xhat, trajectory_up
    ) = build_forward_scatter_geometry(
        z_km=z,
        x_km=x0,
        frequency_hz=frequency_hz,
        meteor_azimuth_deg=meteor_azimuth_deg,
        meteor_elevation_deg=meteor_elevation_deg,
        specular_altitude_km=specular_altitude_km
    )

    # The geometrical specular point is where Tx-P-R has minimum total path.
    specular_z_km = z[specular_index]
    specular_x_km = x0[specular_index]
    specular_beta_deg = np.degrees(beta[specular_index])

    # Multi-Harmonic Atmospheric Wind Shear Field (Fundamental + 3rd + 5th)
    v_fundamental = ratio_fund * np.sin(np.pi * z)
    v_3rd_harmonic = ratio_3rd * np.sin(3.0 * np.pi * z)
    v_5th_harmonic = ratio_5th * np.sin(5.0 * np.pi * z)
    v_wind = wind_speed * (v_fundamental + v_3rd_harmonic + v_5th_harmonic)
    dv_dz = np.gradient(v_wind, dz)

    # Initial entry velocity field (rapid deceleration/ionization ping)
    v_entry = 8.0 * np.sin(2.0 * np.pi * z) + (15.0 * initial_tilt_slope)

    # 3. Continuous Multi-Phase Signal Synthesis Loop
    phase_z = np.zeros(nz)
    echo_signal = np.zeros(num_samples)

    tau_ping_decay = 0.12    # Time constant for entry velocity decay (s)
    tau_shear_growth = 0.40  # Time constant for wind shear onset (s)

    for i in range(num_samples):
        if not active_mask[i]:
            continue

        tau = t_active[i]

        # --- A. CONTINUOUS DOPPLER SHIFT ---
        weight_ping = np.exp(-tau / tau_ping_decay)
        weight_shear = 1.0 - np.exp(-tau / tau_shear_growth)
        
        # Physical horizontal velocity field (m/s).
        v_horizontal = (weight_ping * v_entry) + (weight_shear * v_wind)

        # The model's x-axis is the horizontal Tx-Rx direction.  Convert
        # that velocity to an ECEF vector and project it onto the bistatic
        # Doppler vector u_T + u_R.
        # Reconstruct the local x unit vector from the geometry.
        tx = geodetic_to_ecef(
            DOURBES_LAT_DEG, DOURBES_LON_DEG, DOURBES_ALT_M
        )
        rx = geodetic_to_ecef(
            HUMAIN_LAT_DEG, HUMAIN_LON_DEG, HUMAIN_ALT_M
        )
        ref = geodetic_to_ecef(
            HUMAIN_LAT_DEG, HUMAIN_LON_DEG, 90_000.0
        )
        _, _, up_ref = enu_basis(HUMAIN_LAT_DEG, HUMAIN_LON_DEG)
        horizontal_tx_rx = rx - tx
        horizontal_tx_rx -= np.dot(horizontal_tx_rx, up_ref) * up_ref
        xhat = horizontal_tx_rx / np.linalg.norm(horizontal_tx_rx)

        # Explicit 3-D meteor velocity direction from azimuth/elevation.
        # The evolving scalar speed is represented by v_horizontal.
        el_rad = np.radians(meteor_elevation_deg)
        velocity_ecef = v_horizontal[:, None] * (
            np.cos(el_rad) * trajectory_xhat[None, :]
            + np.sin(el_rad) * trajectory_up[None, :]
        )

        # d(R_T + R_R)/dt = v · (u_T + u_R).
        # Therefore f_D = v · (u_T + u_R) / lambda.
        # Since |u_T + u_R| = 2 cos(beta/2), this is equivalent to
        # f_D = 2 v_b cos(beta/2) / lambda.
        doppler_z = np.sum(
            velocity_ecef * bistatic_bisector, axis=1
        ) / wavelength_m

        # --- B. CONTINUOUS TRAIL DEFORMATION & SPECULAR APERTURE ---
        shear_deformation_time = tau * weight_shear * 0.0025
        dxdz_t = dx0_dz + (dv_dz * shear_deformation_time)

        aperture = 0.005 + (0.015 * weight_shear)
        if tau > t_diffusion_start:
            aperture += 0.010 * (tau - t_diffusion_start)

        # Scale initial specular ping response with alpha_ping
        ping_boost = 1.0 + (alpha_ping * 4.0 * weight_ping)
        specular_weight = ping_boost * np.exp(-(dxdz_t ** 2) / aperture)

        # --- C. PHASE INTEGRATION ---
        inst_freq_z = carrier_freq + doppler_z
        phase_z += 2.0 * np.pi * inst_freq_z * dt

        reflections = specular_weight * np.sin(phase_z)
        echo_signal[i] = np.sum(reflections) * global_envelope[i]

    # Normalize echo signal strength
    if np.max(np.abs(echo_signal)) > 0:
        echo_signal = echo_signal / np.max(np.abs(echo_signal))

    # Add background white noise
    snr_lin = 10 ** (snr_db / 10.0)
    noise = np.random.normal(0, np.sqrt(np.mean(echo_signal**2) / snr_lin), num_samples)

    final_audio = echo_signal + noise
    return (t, final_audio / np.max(np.abs(final_audio)), z, x0, v_wind,
            specular_z_km, specular_x_km, specular_beta_deg)


def main():
    SAMPLE_RATE = 44100
    DURATION = 15.0
    CARRIER_FREQ = 1000.0

    # BRAMS carrier frequency used by the model.
    RADIO_FREQUENCY = 49.97e6  # Hz
    SPECULAR_ALTITUDE_KM = 90.0


    # Initial slider settings
    init_skew  = 12.0
    init_wind  = 80.0
    init_fund  = 1.00
    init_r3rd  = 0.10
    init_r5th  = 0.00
    init_alpha = 0.50
    init_azimuth = 45.0
    init_elevation = 30.0
    init_specular_altitude = 90.0

    # Setup Main Window Layout
    fig = plt.figure(figsize=(16, 9.5), facecolor='black')
    fig.canvas.manager.set_window_title("Overdense Meteor Echo Simulator — Interactive GUI")
    
    # Subplots: 2D Trail Plot (Left) & Spectrogram Plot (Right)
    ax_geom = fig.add_axes([0.07, 0.44, 0.28, 0.50], facecolor='#050515')
    ax_spec = fig.add_axes([0.42, 0.44, 0.53, 0.50], facecolor='navy')

    # Slider Control Axes
    ax_skew  = fig.add_axes([0.18, 0.32, 0.52, 0.022], facecolor='#1f1f1f')
    ax_wind  = fig.add_axes([0.18, 0.27, 0.52, 0.022], facecolor='#1f1f1f')
    ax_fund  = fig.add_axes([0.18, 0.22, 0.52, 0.022], facecolor='#1f1f1f')
    ax_r3rd  = fig.add_axes([0.18, 0.17, 0.52, 0.022], facecolor='#1f1f1f')
    ax_r5th  = fig.add_axes([0.18, 0.12, 0.52, 0.022], facecolor='#1f1f1f')
    ax_alpha = fig.add_axes([0.18, 0.07, 0.52, 0.022], facecolor='#1f1f1f')

    # Control Button Axes
    ax_trigger = fig.add_axes([0.75, 0.22, 0.10, 0.12], facecolor='#2d2d2d')
    ax_play    = fig.add_axes([0.86, 0.22, 0.10, 0.12], facecolor='#2d2d2d')
    ax_save    = fig.add_axes([0.75, 0.08, 0.21, 0.10], facecolor='#2d2d2d')

    # Create Slider Widgets
    s_skew  = Slider(ax_skew,  'Trail Skew (°)',         1.0, 45.0,  valinit=init_skew,  valstep=0.5,  valfmt='%.1f°',   color='cyan')
    s_wind  = Slider(ax_wind,  'Wind Speed (m/s)',       20.0, 150.0, valinit=init_wind,  valstep=1.0,  valfmt='%.0f m/s', color='lime')
    s_fund  = Slider(ax_fund,  'Fundamental Ratio',      0.0, 1.0,   valinit=init_fund,  valstep=0.01, valfmt='%.2f',    color='yellow')
    s_r3rd  = Slider(ax_r3rd,  '3rd Harmonic Ratio',     0.0, 0.5,   valinit=init_r3rd,  valstep=0.01, valfmt='%.2f',    color='orange')
    s_r5th  = Slider(ax_r5th,  '5th Harmonic Ratio',     0.0, 0.5,   valinit=init_r5th,  valstep=0.01, valfmt='%.2f',    color='magenta')
    s_alpha = Slider(ax_alpha, 'Direct Refl. (alpha)',   0.0, 1.0,   valinit=init_alpha, valstep=0.01, valfmt='%.2f',    color='coral')

    # Create Button Widgets
    btn_trigger = Button(ax_trigger, 'Run\nSimulation', color='darkblue', hovercolor='blue')
    btn_trigger.label.set_color('white')

    btn_play = Button(ax_play, 'Play\nAudio', color='darkgreen', hovercolor='green')
    btn_play.label.set_color('white')

    btn_save = Button(ax_save, 'Save Audio File (.wav)', color='#4a2e00', hovercolor='#8b5a00')
    btn_save.label.set_color('white')

    # Apply Styling to Sliders
    for s in [s_skew, s_wind, s_fund, s_r3rd, s_r5th, s_alpha, s_az, s_el]:
        s.label.set_color('white')
        s.valtext.set_color('white')

    current_audio = [None]

    def update_plots(event=None):
        if HAS_SOUNDDEVICE:
            try:
                sd.stop()
            except Exception:
                pass

        skew  = s_skew.val
        wind  = s_wind.val
        fund  = s_fund.val
        r3rd  = s_r3rd.val
        r5th  = s_r5th.val
        alpha = s_alpha.val
        azimuth = s_az.val
        elevation = s_el.val
        specular_altitude = SPECULAR_ALTITUDE_KM

        (
            t, audio, z, x0, v_wind,
            specular_z_km, specular_x_km, specular_beta_deg
        ) = synthesize_meteor_signal(
            skew_deg=skew,
            wind_speed=wind,
            ratio_fund=fund,
            ratio_3rd=r3rd,
            ratio_5th=r5th,
            alpha_ping=alpha,
            meteor_azimuth_deg=azimuth,
            meteor_elevation_deg=elevation,
            specular_altitude_km=specular_altitude,
            frequency_hz=RADIO_FREQUENCY,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
            carrier_freq=CARRIER_FREQ
        )
        current_audio[0] = audio

        # 1. UPDATE 2D SPATIAL TRAIL GEOMETRY PLOT
        ax_geom.clear()
        
        ax_geom.plot(x0, z, color='white', linestyle='--', linewidth=1.5, label='Initial (t=0s)')

        time_steps = [1.5, 3.0, 4.5, 6.0]
        colors = plt.cm.spring(np.linspace(0.2, 1.0, len(time_steps)))

        for tau_step, col in zip(time_steps, colors):
            x_t = x0 + (v_wind * tau_step * 0.0025)
            ax_geom.plot(x_t, z, color=col, linewidth=2.0, label=f't = {tau_step:.1f}s')

        ax_geom.set_title(
            f"Trail Deformation | Az={azimuth:.0f}° El={elevation:.0f}° | "
            f"h_spec={specular_altitude:.1f} km | β={specular_beta_deg:.1f}°",
            color='white', fontsize=10
        )
        ax_geom.set_xlabel("Horizontal Position x (km)", color='white')
        ax_geom.set_ylabel("Relative Altitude z (km)", color='white')
        ax_geom.set_xlim(-1.5, 1.5)
        ax_geom.set_ylim(-1.2, 1.2)
        ax_geom.tick_params(colors='white')

        for spine in ax_geom.spines.values():
            spine.set_color('white')

        ax_geom.grid(True, color='gray', alpha=0.3, linestyle=':')
        legend = ax_geom.legend(loc='upper left', facecolor='#1f1f1f', edgecolor='white', fontsize=8)
        for text in legend.get_texts():
            text.set_color('white')

        # 2. UPDATE SPECTROGRAM PLOT
        ax_spec.clear()
        ax_spec.specgram(
            audio,
            NFFT=8192,
            Fs=SAMPLE_RATE,
            noverlap=7168,
            cmap='jet',
            vmin=-60,
            vmax=-10
        )

        ax_spec.set_title(
            f"Meteor Echo Spectrogram  |  f₀: {RADIO_FREQUENCY/1e6:.2f} MHz  |  "
            f"Skew: {skew:.1f}°  |  Wind: {wind:.0f} m/s  |  "
            f"Fund: {fund:.2f}  |  3rd: {r3rd:.2f}  |  5th: {r5th:.2f}  |  "
            f"α: {alpha:.2f}  |  Az: {azimuth:.0f}°  |  "
            f"El: {elevation:.0f}°  |  h_spec: {specular_altitude:.1f} km  |  "
            f"β_spec: {specular_beta_deg:.1f}°",
            color='white', fontsize=10
        )
        ax_spec.set_xlabel("Time (seconds)", color='white')
        ax_spec.set_ylabel("Doppler Offset (Hz) [Relative to Center Carrier]", color='white')
        ax_spec.set_ylim(CARRIER_FREQ - 120, CARRIER_FREQ + 120)
        ax_spec.set_xlim(0, 12)

        ticks = np.linspace(CARRIER_FREQ - 120, CARRIER_FREQ + 120, 9)
        tick_labels = [f"{int(f - CARRIER_FREQ):+d} Hz" for f in ticks]
        ax_spec.set_yticks(ticks)
        ax_spec.set_yticklabels(tick_labels, color='white')
        ax_spec.tick_params(colors='white')

        for spine in ax_spec.spines.values():
            spine.set_color('white')

        ax_spec.grid(True, color='cyan', alpha=0.18, linestyle='--')
        fig.canvas.draw_idle()

    def play_audio_event(event):
        if HAS_SOUNDDEVICE and current_audio[0] is not None:
            try:
                sd.stop()
                sd.play(np.int16(current_audio[0] * 32767), SAMPLE_RATE, blocking=False)
            except Exception as e:
                print(f"Playback warning: {e}")

    def save_audio_event(event):
        if current_audio[0] is None:
            return

        # Hide root Tkinter window when opening dialog
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAVE Audio", "*.wav"), ("All Files", "*.*")],
            title="Save Meteor Echo Audio"
        )
        
        root.destroy()

        if file_path:
            pcm_16bit = np.int16(current_audio[0] * 32767)
            wav.write(file_path, SAMPLE_RATE, pcm_16bit)
            print(f"Saved audio output to: {file_path}")

    # Attach Button Callbacks
    btn_trigger.on_clicked(update_plots)
    btn_play.on_clicked(play_audio_event)
    btn_save.on_clicked(save_audio_event)

    # Initial plot generation on startup
    update_plots()
    plt.show()


if __name__ == "__main__":
    main()