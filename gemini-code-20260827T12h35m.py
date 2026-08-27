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
    tx = geodetic_to_ecef(tx_lat_deg, tx_lon_deg, tx_alt_m)
    rx = geodetic_to_ecef(rx_lat_deg, rx_lon_deg, rx_alt_m)

    ref_lat = 0.5 * (tx_lat_deg + rx_lat_deg)
    ref_lon = 0.5 * (tx_lon_deg + rx_lon_deg)
    ref = geodetic_to_ecef(ref_lat, ref_lon, specular_altitude_km * 1000.0)
    east, north, up = enu_basis(ref_lat, ref_lon)

    az = np.radians(meteor_azimuth_deg)
    el = np.radians(meteor_elevation_deg)

    xhat = np.sin(az) * east + np.cos(az) * north
    xhat /= np.linalg.norm(xhat)

    trajectory_dir = (
        np.cos(el) * xhat + np.sin(el) * up
    )
    trajectory_dir /= np.linalg.norm(trajectory_dir)

    positions = (
        ref[None, :]
        + (x_km * 1000.0)[:, None] * xhat[None, :]
        + np.zeros((len(z_km), 1))
        + (z_km * 1000.0)[:, None] * up[None, :]
    )

    tx_to_p = positions - tx
    rx_to_p = positions - rx

    r_t = np.linalg.norm(tx_to_p, axis=1)
    r_r = np.linalg.norm(rx_to_p, axis=1)

    u_t = tx_to_p / r_t[:, None]
    u_r = rx_to_p / r_r[:, None]

    cos_beta = np.clip(np.sum(u_t * u_r, axis=1), -1.0, 1.0)
    beta = np.arccos(cos_beta)

    bisector_vector = u_t + u_r
    bisector_norm = np.linalg.norm(bisector_vector, axis=1)
    bisector = bisector_vector / bisector_norm[:, None]

    wavelength = C_LIGHT / frequency_hz

    total_path = r_t + r_r
    specular_index = np.argmin(total_path)

    return (
        positions, tx_to_p, rx_to_p, beta, bisector,
        wavelength, specular_index, total_path,
        trajectory_dir, ref, xhat, up
    )


def synthesize_meteor_signal(
    skew_deg, wind_speed, ratio_fund, ratio_2nd, ratio_3rd,
    meteor_azimuth_deg=45.0,
    meteor_elevation_deg=30.0,
    specular_altitude_km=90.0,
    frequency_hz=49.97e6, sample_rate=44100, duration=15.0,
    carrier_freq=1000.0, snr_db=18.0
):
    num_samples = int(sample_rate * duration)
    dt = 1.0 / sample_rate
    t = np.linspace(0, duration, num_samples, endpoint=False)

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

    specular_z_km = z[specular_index]
    specular_x_km = x0[specular_index]
    specular_beta_deg = np.degrees(beta[specular_index])

    # Multi-Harmonic Atmospheric Wind Shear Field (Fundamental + 2nd + 3rd)
    v_fundamental = ratio_fund * np.sin(np.pi * z)
    v_2nd_harmonic = ratio_2nd * np.sin(2.0 * np.pi * z)
    v_3rd_harmonic = ratio_3rd * np.sin(3.0 * np.pi * z)
    v_wind = wind_speed * (v_fundamental + v_2nd_harmonic + v_3rd_harmonic)
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
        
        v_horizontal = (weight_ping * v_entry) + (weight_shear * v_wind)

        tx = geodetic_to_ecef(DOURBES_LAT_DEG, DOURBES_LON_DEG, DOURBES_ALT_M)
        rx = geodetic_to_ecef(HUMAIN_LAT_DEG, HUMAIN_LON_DEG, HUMAIN_ALT_M)
        _, _, up_ref = enu_basis(HUMAIN_LAT_DEG, HUMAIN_LON_DEG)
        horizontal_tx_rx = rx - tx
        horizontal_tx_rx -= np.dot(horizontal_tx_rx, up_ref) * up_ref

        el_rad = np.radians(meteor_elevation_deg)
        velocity_ecef = v_horizontal[:, None] * (
            np.cos(el_rad) * trajectory_xhat[None, :]
            + np.sin(el_rad) * trajectory_up[None, :]
        )

        doppler_z = np.sum(
            velocity_ecef * bistatic_bisector, axis=1
        ) / wavelength_m

        # --- B. CONTINUOUS TRAIL DEFORMATION & SPECULAR APERTURE ---
        shear_deformation_time = tau * weight_shear * 0.0025
        dxdz_t = dx0_dz + (dv_dz * shear_deformation_time)

        aperture = 0.005 + (0.015 * weight_shear)
        if tau > t_diffusion_start:
            aperture += 0.010 * (tau - t_diffusion_start)

        specular_weight = np.exp(-(dxdz_t ** 2) / aperture)

        # --- C. PHASE INTEGRATION ---
        inst_freq_z = carrier_freq + doppler_z
        phase_z += 2.0 * np.pi * inst_freq_z * dt

        reflections = specular_weight * np.sin(phase_z)
        echo_signal[i] = np.sum(reflections) * global_envelope[i]

    if np.max(np.abs(echo_signal)) > 0:
        echo_signal = echo_signal / np.max(np.abs(echo_signal))

    snr_lin = 10 ** (snr_db / 10.0)
    noise = np.random.normal(0, np.sqrt(np.mean(echo_signal**2) / snr_lin), num_samples)

    final_audio = echo_signal + noise
    return (t, final_audio / np.max(np.abs(final_audio)), z, x0, v_wind,
            specular_z_km, specular_x_km, specular_beta_deg)


def main():
    SAMPLE_RATE = 22050
    DURATION = 15.0
    CARRIER_FREQ = 1000.0

    RADIO_FREQUENCY = 49.97e6  # Hz
    SPECULAR_ALTITUDE_KM = 90.0

    # Initial slider settings
    init_skew  = 12.0
    init_wind  = 80.0
    init_fund  = 1.00
    init_r2nd  = 0.00
    init_r3rd  = 0.10
    init_azimuth = 45.0
    init_elevation = 30.0

    # Setup Main Window Layout
    fig = plt.figure(figsize=(16, 9.5), facecolor='black')
    fig.canvas.manager.set_window_title("Overdense Meteor Echo Simulator — Interactive GUI")
    
    # Adjusted Subplot Layouts: Equal width (0.415 each) covering left and right halves
    ax_geom = fig.add_axes([0.05, 0.44, 0.415, 0.50], facecolor='#050515')
    ax_spec = fig.add_axes([0.535, 0.44, 0.415, 0.50], facecolor='navy')

    # Slider Control Axes
    ax_skew  = fig.add_axes([0.15, 0.340, 0.52, 0.022], facecolor='#1f1f1f')
    ax_wind  = fig.add_axes([0.15, 0.293, 0.52, 0.022], facecolor='#1f1f1f')
    ax_fund  = fig.add_axes([0.15, 0.247, 0.52, 0.022], facecolor='#1f1f1f')
    ax_r2nd  = fig.add_axes([0.15, 0.200, 0.52, 0.022], facecolor='#1f1f1f')
    ax_r3rd  = fig.add_axes([0.15, 0.153, 0.52, 0.022], facecolor='#1f1f1f')
    ax_az    = fig.add_axes([0.15, 0.107, 0.52, 0.022], facecolor='#1f1f1f')
    ax_el    = fig.add_axes([0.15, 0.060, 0.52, 0.022], facecolor='#1f1f1f')

    # Control Button Axes
    ax_trigger  = fig.add_axes([0.72, 0.22, 0.11, 0.12], facecolor='#2d2d2d')
    ax_play     = fig.add_axes([0.84, 0.22, 0.11, 0.12], facecolor='#2d2d2d')
    ax_save_wav = fig.add_axes([0.72, 0.14, 0.23, 0.06], facecolor='#2d2d2d')
    ax_save_png = fig.add_axes([0.72, 0.06, 0.23, 0.06], facecolor='#2d2d2d')

    # Create Slider Widgets (2nd & 3rd Harmonic ranges updated to 0.0 -> 1.0)
    s_skew  = Slider(ax_skew,  'Trail Skew (°)',         1.0, 45.0,  valinit=init_skew,  valstep=0.5,  valfmt='%.1f°',   color='cyan')
    s_wind  = Slider(ax_wind,  'Wind Speed (m/s)',       20.0, 150.0, valinit=init_wind,  valstep=1.0,  valfmt='%.0f m/s', color='lime')
    s_fund  = Slider(ax_fund,  'Fundamental Ratio',      0.0, 1.0,   valinit=init_fund,  valstep=0.01, valfmt='%.2f',    color='yellow')
    s_r2nd  = Slider(ax_r2nd,  '2nd Harmonic Ratio',     0.0, 1.0,   valinit=init_r2nd,  valstep=0.01, valfmt='%.2f',    color='magenta')
    s_r3rd  = Slider(ax_r3rd,  '3rd Harmonic Ratio',     0.0, 1.0,   valinit=init_r3rd,  valstep=0.01, valfmt='%.2f',    color='orange')
    s_az    = Slider(ax_az,    'Azimuth (°)',            0.0, 360.0, valinit=init_azimuth,   valstep=1.0, valfmt='%.0f°', color='deepskyblue')
    s_el    = Slider(ax_el,    'Elevation (°)',          0.0, 90.0,  valinit=init_elevation, valstep=1.0, valfmt='%.0f°', color='violet')

    # Create Button Widgets
    btn_trigger  = Button(ax_trigger, 'Run\nSimulation', color='darkblue', hovercolor='blue')
    btn_trigger.label.set_color('white')

    btn_play     = Button(ax_play, 'Play\nAudio', color='darkgreen', hovercolor='green')
    btn_play.label.set_color('white')

    btn_save_wav = Button(ax_save_wav, 'Save Audio File (.wav)', color='#4a2e00', hovercolor='#8b5a00')
    btn_save_wav.label.set_color('white')

    btn_save_png = Button(ax_save_png, 'Save Spectrogram (.png)', color='#003344', hovercolor='#005577')
    btn_save_png.label.set_color('white')

    # Apply Styling to Sliders
    for s in [s_skew, s_wind, s_fund, s_r2nd, s_r3rd, s_az, s_el]:
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
        r2nd  = s_r2nd.val
        r3rd  = s_r3rd.val
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
            ratio_2nd=r2nd,
            ratio_3rd=r3rd,
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
        
        # Updated horizontal bounds from -3 to 3 km
        ax_geom.set_xlim(-3.0, 3.0)
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
            NFFT=16384,
            Fs=SAMPLE_RATE,
            noverlap=14254,
            cmap='jet',
            vmin=-80,
            vmax=0,
        )

        ax_spec.set_title(
            f"Meteor Echo Spectrogram  |  "
            f"Skew: {skew:.1f}°  |  Wind: {wind:.0f} m/s  |  "
            f"Fund: {fund:.2f}  |  2nd: {r2nd:.2f}  |  3rd: {r3rd:.2f}  |  "
            f"Az: {azimuth:.0f}°  |  El: {elevation:.0f}°",
            color='white', fontsize=10
        )
        ax_spec.set_xlabel("Time (seconds)", color='white')
        ax_spec.set_ylabel("Doppler Offset (Hz) [Relative to Center Carrier]", color='white')
        ax_spec.set_ylim(CARRIER_FREQ - 120, CARRIER_FREQ + 120)
        
        # Preserved 12 seconds x-axis time window
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

    def save_spectrogram_event(event):
        """Export the Spectrogram plot alone as a high-resolution PNG image."""
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
            title="Save Spectrogram Plot Image"
        )

        root.destroy()

        if file_path:
            # Extract bounding box of the spectrogram axes to save only the plot
            extent = ax_spec.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(file_path, bbox_inches=extent.expanded(1.22, 1.25), dpi=300, facecolor=fig.get_facecolor())
            #fig.savefig(file_path, bbox_inches=extent.expanded(1.83, 1.875), dpi=450, facecolor=fig.get_facecolor()) #testing other image format config
            print(f"Saved spectrogram plot image to: {file_path}")

    # Attach Button Callbacks
    btn_trigger.on_clicked(update_plots)
    btn_play.on_clicked(play_audio_event)
    btn_save_wav.on_clicked(save_audio_event)
    btn_save_png.on_clicked(save_spectrogram_event)

    # Initial plot generation on startup
    update_plots()
    plt.show()


if __name__ == "__main__":
    main()