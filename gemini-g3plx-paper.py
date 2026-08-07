import numpy as np
import scipy.io.wavfile as wav

# Ensure GUI backend compatibility on macOS/Conda environments
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


class G3PLXMeteorSimulator:
    def __init__(self, sample_rate=44100, carrier_freq=1000.0, duration=60.0):
        self.fs = sample_rate
        self.f0 = carrier_freq  # 1000 Hz represents 49.970 MHz BRAMS carrier
        self.duration = duration
        self.num_samples = int(self.fs * self.duration)
        self.dt = 1.0 / self.fs
        self.t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        
        # Spatial trail resolution (altitude array along z)
        self.nz = 3500
        self.z = np.linspace(-1.2, 1.2, self.nz)
        self.dz = self.z[1] - self.z[0]

    def _get_envelope_and_timing(self, t_entry=2.0, tau_decay=35.0, trunk_delay=0.5):
        """Calculates plasma lifetime decay and initial trunk delay envelope."""
        active_mask = self.t >= t_entry
        t_active = np.maximum(0.0, self.t - t_entry)

        # Fast ionization attack + slow diffusion decay
        rise = 1.0 - np.exp(-t_active / 0.05)
        decay = np.exp(-t_active / tau_decay)
        envelope = np.zeros(self.num_samples)
        envelope[active_mask] = rise[active_mask] * decay[active_mask]

        # Delay activation for shear deformation (0.5s initial trunk)
        shear_activation = 1.0 / (1.0 + np.exp(-(t_active - trunk_delay) / 0.08))
        
        return active_mask, t_active, envelope, shear_activation

    def simulate_model_1_straight_trail(self):
        """
        MODEL 1: Straight-line point-reflector trail model.
        Specular condition holds at a single location; produces an unbranched burst.
        """
        active_mask, t_active, envelope, _ = self._get_envelope_and_timing()
        
        # Uniform velocity along a straight column
        v_wind = 0.0  # Stationary straight trail
        dx0_dz = np.gradient(0.01 * self.z, self.dz)
        
        phase_z = np.zeros(self.nz)
        I_signal = np.zeros(self.num_samples)

        for i in range(self.num_samples):
            if not active_mask[i]:
                continue
            
            # Specular aperture for a straight line
            specular_weight = np.exp(-(dx0_dz ** 2) / 0.015)
            
            # Phase accumulation at carrier frequency
            phase_z += 2.0 * np.pi * self.f0 * self.dt
            I_signal[i] = np.sum(specular_weight * np.cos(phase_z)) * envelope[i]

        return I_signal

    def simulate_model_2_sinusoidal_c_shape(self):
        """
        MODEL 2: Trail deformed by a fundamental sinusoidal wind profile.
        Generates specular splitting into a classic C-shape echo.
        """
        active_mask, t_active, envelope, shear_act = self._get_envelope_and_timing()

        # Fundamental 1st Harmonic Wind Shear Profile
        v_wind = 80.0 * np.sin(np.pi * self.z)
        dx0_dz = np.gradient(0.02 * self.z, self.dz)
        dv_dz = np.gradient(v_wind, self.dz)

        phase_z = np.zeros(self.nz)
        I_signal = np.zeros(self.num_samples)

        for i in range(self.num_samples):
            if not active_mask[i]:
                continue

            tau = t_active[i]
            shear_factor = 0.12 * (1.0 - np.exp(-tau / 25.0)) * shear_act[i]
            dxdz_t = dx0_dz + (dv_dz * shear_factor)

            # Specular point selection
            aperture = 0.018 + (0.0004 * tau)
            specular_weight = np.exp(-(dxdz_t ** 2) / aperture)

            # Local Doppler shift integration
            inst_freq = self.f0 + v_wind
            phase_z += 2.0 * np.pi * inst_freq * self.dt

            I_signal[i] = np.sum(specular_weight * np.cos(phase_z)) * envelope[i]

        return I_signal

    def simulate_model_3_epsilon_harmonic(self):
        """
        MODEL 3: Trail deformed by fundamental + 3rd harmonic wind shear.
        Generates multi-branch specular points that cross to form the Epsilon shape.
        """
        active_mask, t_active, envelope, shear_act = self._get_envelope_and_timing()

        # 1st + 3rd Harmonic Deformation Profile (G3PLX Epsilon Equation)
        v_wind = 80.0 * (np.sin(np.pi * self.z) + 0.40 * np.sin(3.0 * np.pi * self.z))
        dx0_dz = np.gradient(0.02 * self.z, self.dz)
        dv_dz = np.gradient(v_wind, self.dz)

        phase_z = np.zeros(self.nz)
        I_signal = np.zeros(self.num_samples)

        for i in range(self.num_samples):
            if not active_mask[i]:
                continue

            tau = t_active[i]
            shear_factor = 0.12 * (1.0 - np.exp(-tau / 25.0)) * shear_act[i]
            dxdz_t = dx0_dz + (dv_dz * shear_factor)

            aperture = 0.018 + (0.0004 * tau)
            specular_weight = np.exp(-(dxdz_t ** 2) / aperture)

            inst_freq = self.f0 + v_wind
            phase_z += 2.0 * np.pi * inst_freq * self.dt

            I_signal[i] = np.sum(specular_weight * np.cos(phase_z)) * envelope[i]

        return I_signal

    def add_beacon_and_noise(self, echo_signal, snr_db=18.0):
        """Adds low-power narrow beacon carrier (0 Hz offset) and background AWGN."""
        if np.max(np.abs(echo_signal)) > 0:
            echo_signal = echo_signal / np.max(np.abs(echo_signal))

        # Direct BRAMS Beacon Line
        beacon = 0.035 * np.sin(2.0 * np.pi * self.f0 * self.t)

        total = echo_signal + beacon
        snr_lin = 10 ** (snr_db / 10.0)
        noise = np.random.normal(0, np.sqrt(np.mean(total**2) / snr_lin), self.num_samples)

        final_sig = total + noise
        return final_sig / np.max(np.abs(final_sig))


def main():
    SAMPLE_RATE = 44100
    DURATION = 60.0
    CARRIER_FREQ = 1000.0

    sim = G3PLXMeteorSimulator(sample_rate=SAMPLE_RATE, carrier_freq=CARRIER_FREQ, duration=DURATION)

    print("Generating Progressive G3PLX Meteor Models...")
    sig1 = sim.add_beacon_and_noise(sim.simulate_model_1_straight_trail())
    sig2 = sim.add_beacon_and_noise(sim.simulate_model_2_sinusoidal_c_shape())
    sig3 = sim.add_beacon_and_noise(sim.simulate_model_3_epsilon_harmonic())

    # Save Epsilon audio file to disk
    wav_file = "g3plx_epsilon_complete.wav"
    wav.write(wav_file, SAMPLE_RATE, np.int16(sig3 * 32767))
    print(f"Saved audio output to '{wav_file}'")

    if HAS_SOUNDDEVICE:
        sd.play(np.int16(sig3 * 32767), SAMPLE_RATE)

    # Plot Comparison Spectrogram (3 Stages)
    plt.figure(figsize=(15, 10), facecolor='black')
    
    models = [
        ("Stage 1: Straight-Line Trail Model (Single Specular Region)", sig1),
        ("Stage 2: Sinusoidal Wind Shear (C-Shape Branch Evolution)", sig2),
        ("Stage 3: 1st + 3rd Harmonic Wind Shear (Multi-Branch Epsilon Shape)", sig3)
    ]

    for idx, (title, audio) in enumerate(models, start=1):
        ax = plt.subplot(3, 1, idx)
        ax.set_facecolor('navy')

        plt.specgram(
            audio,
            NFFT=8192,
            Fs=SAMPLE_RATE,
            noverlap=7168,
            cmap='jet',
            vmin=-60,
            vmax=-10
        )

        plt.title(title, color='white', fontsize=11)
        plt.ylabel("Doppler Offset (Hz)", color='white')
        plt.ylim(CARRIER_FREQ - 100, CARRIER_FREQ + 100)

        ticks = np.linspace(CARRIER_FREQ - 100, CARRIER_FREQ + 100, 5)
        tick_labels = [f"{int(f - CARRIER_FREQ):+d} Hz" for f in ticks]
        plt.yticks(ticks, tick_labels, color='white')

        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        plt.grid(True, color='cyan', alpha=0.15, linestyle='--')

    plt.xlabel("Time (seconds)", color='white')
    plt.tight_layout()

    img_filename = "g3plx_progressive_stages.png"
    plt.savefig(img_filename, dpi=300, facecolor='black', edgecolor='none')
    print(f"Saved comparison spectrogram to '{img_filename}'")

    plt.show()


if __name__ == "__main__":
    main()