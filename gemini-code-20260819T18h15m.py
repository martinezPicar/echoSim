import numpy as np
import scipy.io.wavfile as wav
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


def synthesize_meteor_signal(skew_deg, wind_speed, ratio_fund, ratio_3rd, ratio_5th, alpha_ping, sample_rate=44100, duration=15.0, carrier_freq=1000.0, snr_db=18.0):
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
        
        doppler_z = (weight_ping * v_entry) + (weight_shear * v_wind)

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
    return t, final_audio / np.max(np.abs(final_audio)), z, x0, v_wind


def main():
    SAMPLE_RATE = 44100
    DURATION = 15.0
    CARRIER_FREQ = 1000.0

    # Initial slider settings
    init_skew  = 12.0
    init_wind  = 80.0
    init_fund  = 1.00
    init_r3rd  = 0.10
    init_r5th  = 0.00
    init_alpha = 0.50

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
    btn_trigger = Button(ax_trigger, '⚡ Trigger\nSimulation', color='darkblue', hovercolor='blue')
    btn_trigger.label.set_color('white')

    btn_play = Button(ax_play, '▶ Play\nAudio', color='darkgreen', hovercolor='green')
    btn_play.label.set_color('white')

    btn_save = Button(ax_save, '💾 Save Audio File (.wav)', color='#4a2e00', hovercolor='#8b5a00')
    btn_save.label.set_color('white')

    # Apply Styling to Sliders
    for s in [s_skew, s_wind, s_fund, s_r3rd, s_r5th, s_alpha]:
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

        t, audio, z, x0, v_wind = synthesize_meteor_signal(
            skew_deg=skew,
            wind_speed=wind,
            ratio_fund=fund,
            ratio_3rd=r3rd,
            ratio_5th=r5th,
            alpha_ping=alpha,
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

        ax_geom.set_title("2D Trail Spatial Deformation", color='white', fontsize=11)
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
            f"Meteor Echo Spectrogram  |  Skew: {skew:.1f}°  |  Wind: {wind:.0f}m/s  |  Fund: {fund:.2f}  |  3rd: {r3rd:.2f}  |  5th: {r5th:.2f}  |  alpha: {alpha:.2f}",
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