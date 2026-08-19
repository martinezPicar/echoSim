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


def synthesize_meteor_signal(skew_deg, wind_speed, ratio_3rd, sample_rate=44100, duration=15.0, carrier_freq=1000.0, snr_db=18.0):
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

    # Multi-Harmonic Atmospheric Wind Shear Field
    v_fundamental = np.sin(np.pi * z)
    v_3rd_harmonic = ratio_3rd * np.sin(3.0 * np.pi * z)
    v_wind = wind_speed * (v_fundamental + v_3rd_harmonic)
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
        # Smoothly blend entry ping velocity into wind shear velocity
        weight_ping = 0.0 * np.exp(-tau / tau_ping_decay) # 1.0
        weight_shear = 1.0 - np.exp(-tau / tau_shear_growth)
        
        doppler_z = (weight_ping * v_entry) + (weight_shear * v_wind)

        # --- B. CONTINUOUS TRAIL DEFORMATION & SPECULAR APERTURE ---
        # Mechanical bending deformation scales smoothly with time
        shear_deformation_time = tau * weight_shear * 0.0025
        dxdz_t = dx0_dz + (dv_dz * shear_deformation_time)

        # Continuous aperture broadening (from tight specular line to broader reflection)
        aperture = 0.005 + (0.015 * weight_shear)
        if tau > t_diffusion_start:
            aperture += 0.010 * (tau - t_diffusion_start)

        # Specular weight calculation (Gaussian alignment along trail)
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
    return t, final_audio / np.max(np.abs(final_audio))


def main():
    SAMPLE_RATE = 44100
    DURATION = 15.0
    CARRIER_FREQ = 1000.0

    init_skew = 12.0
    init_wind = 80.0
    init_ratio = 0.10

    fig = plt.figure(figsize=(14, 8), facecolor='black')
    fig.canvas.manager.set_window_title("BRAMS Meteor Echo Simulator — Smooth Dynamic Transition")
    
    ax_spec = fig.add_axes([0.1, 0.32, 0.82, 0.60], facecolor='navy')

    ax_skew  = fig.add_axes([0.18, 0.20, 0.65, 0.03], facecolor='#1f1f1f')
    ax_wind  = fig.add_axes([0.18, 0.14, 0.65, 0.03], facecolor='#1f1f1f')
    ax_ratio = fig.add_axes([0.18, 0.08, 0.65, 0.03], facecolor='#1f1f1f')
    ax_play  = fig.add_axes([0.85, 0.12, 0.08, 0.08], facecolor='#2d2d2d')

    s_skew = Slider(ax_skew, 'Trail Skew (°)', 1.0, 45.0, valinit=init_skew, valstep=0.5, valfmt='%.1f°', color='cyan')
    s_wind = Slider(ax_wind, 'Wind Speed (m/s)', 20.0, 150.0, valinit=init_wind, valstep=1.0, valfmt='%.0f m/s', color='lime')
    s_ratio = Slider(ax_ratio, '3rd Harmonic Ratio', 0.0, 0.5, valinit=init_ratio, valstep=0.01, valfmt='%.2f', color='orange')
    btn_play = Button(ax_play, 'Play\nAudio', color='darkgreen', hovercolor='green')

    for s in [s_skew, s_wind, s_ratio]:
        s.label.set_color('white')
        s.valtext.set_color('white')

    current_audio = [None]

    def update_spectrogram(val=None):
        if HAS_SOUNDDEVICE:
            try:
                sd.stop()
            except Exception:
                pass

        skew = s_skew.val
        wind = s_wind.val
        ratio = s_ratio.val

        t, audio = synthesize_meteor_signal(
            skew_deg=skew,
            wind_speed=wind,
            ratio_3rd=ratio,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
            carrier_freq=CARRIER_FREQ
        )
        current_audio[0] = audio

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
            f"BRAMS Meteor Echo (Smooth Continuous Transition)  |  Skew: {skew:.1f}°  |  Wind: {wind:.0f} m/s  |  3rd Harmonic Ratio: {ratio:.2f}",
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

    s_skew.on_changed(update_spectrogram)
    s_wind.on_changed(update_spectrogram)
    s_ratio.on_changed(update_spectrogram)
    btn_play.on_clicked(play_audio_event)

    update_spectrogram()
    plt.show()


if __name__ == "__main__":
    main()