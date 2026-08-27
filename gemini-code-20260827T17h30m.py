import numpy as np
import scipy.io.wavfile as wav
import tkinter as tk
from tkinter import filedialog

import matplotlib
matplotlib.use('TkAgg')  # GUI backend required for interactive controls
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


# ---------------------------------------------------------------------------
# Generalized forward-scatter geometry
# ---------------------------------------------------------------------------
C_LIGHT = 299_792_458.0


def build_forward_scatter_geometry(
    z_km, x_km, frequency_hz,
    distance_km=50.0,
    meteor_azimuth_deg=45.0,
    meteor_elevation_deg=30.0,
    specular_altitude_km=90.0,
):
    """
    Build local forward-scatter geometry parameterized by Tx-Rx distance.
    Tx is placed at (-distance/2, 0, 0) and Rx at (+distance/2, 0, 0) relative
    to the specular center point at altitude specular_altitude_km.
    """
    az = np.radians(meteor_azimuth_deg)
    el = np.radians(meteor_elevation_deg)

    # Local ENU basis vectors
    east = np.array([1.0, 0.0, 0.0])
    north = np.array([0.0, 1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])

    # Horizontal trajectory direction derived from azimuth
    xhat = np.sin(az) * east + np.cos(az) * north
    xhat /= np.linalg.norm(xhat)

    # Full 3D meteor trajectory direction vector
    trajectory_dir = np.cos(el) * xhat + np.sin(el) * up
    trajectory_dir /= np.linalg.norm(trajectory_dir)

    # Center reference point at specular altitude
    ref = np.array([0.0, 0.0, specular_altitude_km * 1000.0])

    # Tx and Rx positions set symmetrically around origin based on baseline distance
    tx = np.array([- (distance_km * 1000.0) / 2.0, 0.0, 0.0])
    rx = np.array([+ (distance_km * 1000.0) / 2.0, 0.0, 0.0])

    positions = (
        ref[None, :]
        + (x_km * 1000.0)[:, None] * xhat[None, :]
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
    distance_km=50.0,
    meteor_azimuth_deg=45.0,
    meteor_elevation_deg=30.0,
    specular_altitude_km=90.0,
    frequency_hz=50.0e6, sample_rate=22050, duration=15.0,
    carrier_freq=1000.0, snr_db=18.0, add_noise=True
):
    num_samples = int(sample_rate * duration)
    dt = 1.0 / sample_rate
    t = np.linspace(0, duration, num_samples, endpoint=False)

    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    wavelength_m = C_LIGHT / frequency_hz

    # 1. Event Timing & Delayed Diffusion Envelope
    t_entry = 2.0
    active_mask = t >= t_entry
    t_active = np.maximum(0.0, t - t_entry)

    rise_env = 1.0 - np.exp(-t_active / 0.005)
    t_diffusion_start = 3.0
    diffusion_time = np.maximum(0.0, t_active - t_diffusion_start)
    decay_env = np.exp(-diffusion_time / 1.0)
    
    global_envelope = np.zeros(num_samples)
    global_envelope[active_mask] = rise_env[active_mask] * decay_env[active_mask]

    # 2. Altitude Layer Definition & Spatial Profile
    nz = 4000
    z = np.linspace(-1.2, 1.2, nz)
    dz = z[1] - z[0]

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
        distance_km=distance_km,
        meteor_azimuth_deg=meteor_azimuth_deg,
        meteor_elevation_deg=meteor_elevation_deg,
        specular_altitude_km=specular_altitude_km
    )

    specular_z_km = z[specular_index]
    specular_x_km = x0[specular_index]
    specular_beta_deg = np.degrees(beta[specular_index])

    # Multi-Harmonic Wind Shear Field
    v_fundamental = ratio_fund * np.sin(np.pi * z)
    v_2nd_harmonic = ratio_2nd * np.sin(2.0 * np.pi * z)
    v_3rd_harmonic = ratio_3rd * np.sin(3.0 * np.pi * z)
    v_wind = wind_speed * (v_fundamental + v_2nd_harmonic + v_3rd_harmonic)
    dv_dz = np.gradient(v_wind, dz)

    v_entry = 8.0 * np.sin(2.0 * np.pi * z) + (15.0 * initial_tilt_slope)

    # 3. Continuous Multi-Phase Signal Synthesis Loop
    phase_z = np.zeros(nz)
    echo_signal = np.zeros(num_samples)

    tau_ping_decay = 0.12
    tau_shear_growth = 0.40

    for i in range(num_samples):
        if not active_mask[i]:
            continue

        tau = t_active[i]

        weight_ping = np.exp(-tau / tau_ping_decay)
        weight_shear = 1.0 - np.exp(-tau / tau_shear_growth)
        
        v_horizontal = (weight_ping * v_entry) + (weight_shear * v_wind)

        el_rad = np.radians(meteor_elevation_deg)
        velocity_ecef = v_horizontal[:, None] * (
            np.cos(el_rad) * trajectory_xhat[None, :]
            + np.sin(el_rad) * trajectory_up[None, :]
        )

        doppler_z = np.sum(
            velocity_ecef * bistatic_bisector, axis=1
        ) / wavelength_m

        shear_deformation_time = tau * weight_shear * 0.0025
        dxdz_t = dx0_dz + (dv_dz * shear_deformation_time)

        aperture = 0.005 + (0.015 * weight_shear)
        if tau > t_diffusion_start:
            aperture += 0.010 * (tau - t_diffusion_start)

        specular_weight = np.exp(-(dxdz_t ** 2) / aperture)

        inst_freq_z = carrier_freq + doppler_z
        phase_z += 2.0 * np.pi * inst_freq_z * dt

        reflections = specular_weight * np.sin(phase_z)
        echo_signal[i] = np.sum(reflections) * global_envelope[i]

    if np.max(np.abs(echo_signal)) > 0:
        echo_signal = echo_signal / np.max(np.abs(echo_signal))

    if add_noise:
        snr_lin = 10 ** (snr_db / 10.0)
        noise = np.random.normal(0, np.sqrt(np.mean(echo_signal**2) / snr_lin), num_samples)
        final_audio = echo_signal + noise
    else:
        final_audio = echo_signal

    return (t, final_audio / np.max(np.abs(final_audio)), z, x0, v_wind,
            specular_z_km, specular_x_km, specular_beta_deg)


def main():
    SAMPLE_RATE = 22050
    DURATION = 15.0
    CARRIER_FREQ = 1000.0

    # Initial slider settings
    init_dist  = 50.0   # km (20 to 200)
    init_hspec = 90.0   # km (85 to 130)
    init_freq  = 50.0   # MHz (30 to 300)

    init_skew  = 12.0
    init_wind  = 80.0
    init_fund  = 1.00
    init_r2nd  = 0.00
    init_r3rd  = 0.10
    init_azimuth = 180.0
    init_elevation = 45.0

    fig = plt.figure(figsize=(16, 10), facecolor='black')
    fig.canvas.manager.set_window_title("Overdense Meteor Echo Simulator — Interactive GUI")
    
    # Half-and-half equal-width subplots
    ax_geom = fig.add_axes([0.05, 0.48, 0.415, 0.46], facecolor='#050515')
    ax_spec = fig.add_axes([0.535, 0.48, 0.415, 0.46], facecolor='navy')

    # Slider Control Axes (10 Sliders total)
    ax_dist  = fig.add_axes([0.15, 0.400, 0.52, 0.018], facecolor='#1f1f1f')
    ax_hspec = fig.add_axes([0.15, 0.365, 0.52, 0.018], facecolor='#1f1f1f')
    ax_freq  = fig.add_axes([0.15, 0.330, 0.52, 0.018], facecolor='#1f1f1f')
    ax_skew  = fig.add_axes([0.15, 0.295, 0.52, 0.018], facecolor='#1f1f1f')
    ax_wind  = fig.add_axes([0.15, 0.260, 0.52, 0.018], facecolor='#1f1f1f')
    ax_fund  = fig.add_axes([0.15, 0.225, 0.52, 0.018], facecolor='#1f1f1f')
    ax_r2nd  = fig.add_axes([0.15, 0.190, 0.52, 0.018], facecolor='#1f1f1f')
    ax_r3rd  = fig.add_axes([0.15, 0.155, 0.52, 0.018], facecolor='#1f1f1f')
    ax_az    = fig.add_axes([0.15, 0.120, 0.52, 0.018], facecolor='#1f1f1f')
    ax_el    = fig.add_axes([0.15, 0.085, 0.52, 0.018], facecolor='#1f1f1f')

    # Control Button & Selector Axes
    ax_noise    = fig.add_axes([0.72, 0.370, 0.23, 0.05], facecolor='#1f1f1f')
    ax_trigger  = fig.add_axes([0.72, 0.24, 0.11, 0.12], facecolor='#2d2d2d')
    ax_play     = fig.add_axes([0.84, 0.24, 0.11, 0.12], facecolor='#2d2d2d')
    ax_save_wav = fig.add_axes([0.72, 0.16, 0.23, 0.06], facecolor='#2d2d2d')
    ax_save_png = fig.add_axes([0.72, 0.08, 0.23, 0.06], facecolor='#2d2d2d')

    # Slider Instantiations
    s_dist  = Slider(ax_dist,  'Tx-Rx Distance (km)', 20.0, 200.0, valinit=init_dist,  valstep=1.0,  valfmt='%.0f km', color='lightgreen')
    s_hspec = Slider(ax_hspec, 'Specular Alt (km)',  85.0, 130.0, valinit=init_hspec, valstep=0.5,  valfmt='%.1f km', color='cyan')
    s_freq  = Slider(ax_freq,  'Carrier Frequency (MHz)',     30.0, 300.0, valinit=init_freq,  valstep=0.1,  valfmt='%.1f MHz', color='yellow')
    s_skew  = Slider(ax_skew,  'Trail Skew (°)',      1.0,  45.0,  valinit=init_skew,  valstep=0.5,  valfmt='%.1f°',   color='coral')
    s_wind  = Slider(ax_wind,  'Wind Speed (m/s)',    20.0, 150.0, valinit=init_wind,  valstep=1.0,  valfmt='%.0f m/s', color='lime')
    s_fund  = Slider(ax_fund,  'Fundamental Ratio',   0.0,  1.0,   valinit=init_fund,  valstep=0.01, valfmt='%.2f',    color='gold')
    s_r2nd  = Slider(ax_r2nd,  '2nd Harmonic Ratio',  0.0,  1.0,   valinit=init_r2nd,  valstep=0.01, valfmt='%.2f',    color='magenta')
    s_r3rd  = Slider(ax_r3rd,  '3rd Harmonic Ratio',  0.0,  1.0,   valinit=init_r3rd,  valstep=0.01, valfmt='%.2f',    color='orange')
    s_az    = Slider(ax_az,    'Azimuth (°)',         0.0,  360.0, valinit=init_azimuth,valstep=1.0, valfmt='%.0f°', color='deepskyblue')
    s_el    = Slider(ax_el,    'Elevation (°)',       0.0,  90.0,  valinit=init_elevation,valstep=1.0,valfmt='%.0f°', color='violet')

    # Radio Button Selector Instantiation
    radio_noise = RadioButtons(ax_noise, ('Noise ON', 'Noise OFF'), active=0, activecolor='cyan')
    for label in radio_noise.labels:
        label.set_color('white')
        label.set_fontsize(9)

    # Button Instantiations
    btn_trigger  = Button(ax_trigger, 'Run\nSimulation', color='darkblue', hovercolor='blue')
    btn_trigger.label.set_color('white')

    btn_play     = Button(ax_play, 'Play\nAudio', color='darkgreen', hovercolor='green')
    btn_play.label.set_color('white')

    btn_save_wav = Button(ax_save_wav, 'Save Audio File (.wav)', color='#4a2e00', hovercolor='#8b5a00')
    btn_save_wav.label.set_color('white')

    btn_save_png = Button(ax_save_png, 'Save Spectrogram (.png)', color='#003344', hovercolor='#005577')
    btn_save_png.label.set_color('white')

    # Apply Styling to Sliders
    for s in [s_dist, s_hspec, s_freq, s_skew, s_wind, s_fund, s_r2nd, s_r3rd, s_az, s_el]:
        s.label.set_color('white')
        s.valtext.set_color('white')

    current_audio = [None]

    def update_plots(event=None):
        if HAS_SOUNDDEVICE:
            try:
                sd.stop()
            except Exception:
                pass

        dist_km   = s_dist.val
        hspec_km  = s_hspec.val
        freq_mhz  = s_freq.val
        skew      = s_skew.val
        wind      = s_wind.val
        fund      = s_fund.val
        r2nd      = s_r2nd.val
        r3rd      = s_r3rd.val
        azimuth   = s_az.val
        elevation = s_el.val

        use_noise = (radio_noise.value_selected == 'Noise ON')

        (
            t, audio, z, x0, v_wind,
            specular_z_km, specular_x_km, specular_beta_deg
        ) = synthesize_meteor_signal(
            skew_deg=skew,
            wind_speed=wind,
            ratio_fund=fund,
            ratio_2nd=r2nd,
            ratio_3rd=r3rd,
            distance_km=dist_km,
            meteor_azimuth_deg=azimuth,
            meteor_elevation_deg=elevation,
            specular_altitude_km=hspec_km,
            frequency_hz=freq_mhz * 1e6,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
            carrier_freq=CARRIER_FREQ,
            add_noise=use_noise
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
            f"Trail Deformation | $d$={dist_km:.0f} km | $f_0$={freq_mhz:.1f} MHz | "
            f"$h_{{\\mathrm{{spec}}}}$={hspec_km:.1f} km | $\\beta$={specular_beta_deg:.1f}°",
            color='white', fontsize=10
        )
        ax_geom.set_xlabel("Horizontal Position x (km)", color='white')
        ax_geom.set_ylabel("Relative Altitude z (km)", color='white')
        
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
            f"Spectrogram | $f_0$={freq_mhz:.1f} MHz | $d$={dist_km:.0f} km | $h_{{\\mathrm{{spec}}}}$={hspec_km:.0f} km | "
            f"Az:{azimuth:.0f}° El:{elevation:.0f}°",
            color='white', fontsize=10
        )
        ax_spec.set_xlabel("Time (seconds)", color='white')
        ax_spec.set_ylabel("Doppler Offset (Hz) [Relative to Carrier]", color='white')
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
        """Export the Spectrogram plot alone as a high-resolution PNG image (600 DPI)."""
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
            title="Save Spectrogram Plot Image"
        )
    
        root.destroy()

        if file_path:
            extent = ax_spec.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(file_path, bbox_inches=extent.expanded(1.22, 1.25), dpi=600, facecolor=fig.get_facecolor())
            print(f"Saved high-resolution spectrogram plot image to: {file_path}")

    # Attach Callbacks
    radio_noise.on_clicked(update_plots)
    btn_trigger.on_clicked(update_plots)
    btn_play.on_clicked(play_audio_event)
    btn_save_wav.on_clicked(save_audio_event)
    btn_save_png.on_clicked(save_spectrogram_event)

    # Initial plot generation on startup
    update_plots()
    plt.show()


if __name__ == "__main__":
    main()