import numpy as np
import scipy.io.wavfile as wav

import matplotlib
matplotlib.use('TkAgg')  # GUI backend required for interactive sliders
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


def synthesize_meteor_signal(skew_deg, wind_speed, ratio_3rd, ratio_5th, sample_rate=44100, duration=15.0, carrier_freq=1000.0, snr_db=18.0):
    num_samples = int(sample_rate * duration)
    dt = 1.0 / sample_rate
    t = np.linspace(0, duration, num_samples, endpoint=False)

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

    # Multi-Harmonic Atmospheric Wind Shear Field (Fundamental + 3rd + 5th)
    v_fundamental = np.sin(np.pi * z)
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
        
        doppler_z = (weight_ping * v_entry) + (weight_shear * v_wind)

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

    # Normalize echo signal strength
    if np.max(np.abs(echo_signal)) > 0:
        echo_signal = echo_signal / np.max(np.abs(echo_signal))

    # Add background white noise
    snr_lin = 10 ** (snr_db / 10.0)
    noise = np.random.normal(0, np.sqrt(np.mean(echo_signal**2) / snr_lin), num_samples)

    final_audio = echo_signal + noise
    return t, final_audio / np.max(np.abs(final_audio)), z, x0, v_wind


def main():
    SAMPLE_RATE = 44100
    DURATION = 15.0
    CARRIER_FREQ = 1000.0

    # Initial slider settings
    init_skew  = 12.0
    init_wind  = 80.0
    init_r3rd  = 0.10
    init_r5th  = 0.0

    # Setup Main Window Layout
    fig = plt.figure(figsize=(16, 9), facecolor='black')
    fig.canvas.manager.set_window_title("Meteor Echo Simulator — 2D Trail Deformation & Spectrogram")
    
    # Left Subplot: 2D Spatial Trail Geometry Plot
    ax_geom = fig.add_axes([0.07, 0.38, 0.28, 0.55], facecolor='#050515')

    # Right Subplot: Spectrogram Plot
    ax_spec = fig.add_axes([0.42, 0.38, 0.53, 0.55], facecolor='navy')

    # Slider Control Axes
    ax_skew  = fig.add_axes([0.18, 0.24, 0.58, 0.025], facecolor='#1f1f1f')
    ax_wind  = fig.add_axes([0.18, 0.18, 0.58, 0.025], facecolor='#1f1f1f')
    ax_r3rd  = fig.add_axes([0.18, 0.12, 0.58, 0.025], facecolor='#1f1f1f')
    ax_r5th  = fig.add_axes([0.18, 0.06, 0.58, 0.025], facecolor='#1f1f1f')

    # Play Button Axis
    ax_play  = fig.add_axes([0.83, 0.12, 0.10, 0.10], facecolor='#2d2d2d')

    # Create Slider Widgets
    s_skew = Slider(ax_skew, 'Trail Skew (°)',     1.0, 45.0,  valinit=init_skew, valstep=0.5,  valfmt='%.1f°',   color='cyan')
    s_wind = Slider(ax_wind, 'Wind Speed (m/s)',   20.0, 150.0, valinit=init_wind, valstep=1.0,  valfmt='%.0f m/s', color='lime')
    s_r3rd = Slider(ax_r3rd, '3rd Harmonic Ratio', 0.0, 0.5,   valinit=init_r3rd, valstep=0.01, valfmt='%.2f',    color='orange')
    s_r5th = Slider(ax_r5th, '5th Harmonic Ratio', 0.0, 0.5,   valinit=init_r5th, valstep=0.01, valfmt='%.2f',    color='magenta')

    btn_play = Button(ax_play, 'Play\nAudio', color='darkgreen', hovercolor='green')
    btn_play.label.set_color('white')

    # Apply Styling to Sliders
    for s in [s_skew, s_wind, s_r3rd, s_r5th]:
        s.label.set_color('white')
        s.valtext.set_color('white')

    current_audio = [None]

    def update_plots(val=None):
        if HAS_SOUNDDEVICE:
            try:
                sd.stop()
            except Exception:
                pass

        skew = s_skew.val
        wind = s_wind.val
        r3rd = s_r3rd.val
        r5th = s_r5th.val

        t, audio, z, x0, v_wind = synthesize_meteor_signal(
            skew_deg=skew,
            wind_speed=wind,
            ratio_3rd=r3rd,
            ratio_5th=r5th,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
            carrier_freq=CARRIER_FREQ
        )
        current_audio[0] = audio

        # -------------------------------------------------------------
        # 1. UPDATE 2D SPATIAL TRAIL GEOMETRY PLOT
        # -------------------------------------------------------------
        ax_geom.clear()
        
        # Plot initial trail (t = 0s)
        ax_geom.plot(x0, z, color='white', linestyle='--', linewidth=1.5, label='Initial (t=0s)')

        # Draw evolving dynamic shapes at time steps (t = 1.5s, 3.0s, 4.5s, 6.0s)
        time_steps = [1.5, 3.0, 4.5, 6.0]
        colors = plt.cm.spring(np.linspace(0.2, 1.0, len(time_steps)))

        for tau_step, col in zip(time_steps, colors):
            x_t = x0 + (v_wind * tau_step * 0.0025)
            ax_geom.plot(x_t, z, color=col, linewidth=2.0, label=f't = {tau_step:.1f}s')

        ax_geom.set_title("2D Trail Deformation", color='white', fontsize=11)
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

        # -------------------------------------------------------------
        # 2. UPDATE SPECTROGRAM PLOT
        # -------------------------------------------------------------
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
            f"Meteor Echo Spectrogram  |  Skew: {skew:.1f}°  |  Wind: {wind:.0f} m/s  |  3rd: {r3rd:.2f}  |  5th: {r5th:.2f}",
            color='white', fontsize=11
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

    # Attach Callbacks
    s_skew.on_changed(update_plots)
    s_wind.on_changed(update_plots)
    s_r3rd.on_changed(update_plots)
    s_r5th.on_changed(update_plots)
    btn_play.on_clicked(play_audio_event)

    update_plots()
    plt.show()


if __name__ == "__main__":
    main()