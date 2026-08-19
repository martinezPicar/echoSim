import io
import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk
import wave

# Force Matplotlib to use TkAgg
import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# Audio playback library setup (using PyAudio or simpleaudio if available, or wave export)
try:
    import simpleaudio as sa
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


class MeteorEpsilonEchoSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Meteor Epsilon Echo Simulator")
        self.root.geometry("900x750")

        # Communication queue between worker thread and main GUI thread
        self.queue = queue.Queue()

        # State storage for audio and plot data
        self.last_audio_data = None
        self.sample_rate = 44100

        self._setup_ui()
        self._poll_queue()

    def _setup_ui(self):
        # --- Control Panel (Sliders & Parameters) ---
        control_frame = ttk.LabelFrame(self.root, text="Echo Parameters", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Slider 1: Epsilon Factor
        ttk.Label(control_frame, text="Epsilon Factor (ε):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.epsilon_var = tk.DoubleVar(value=0.85)
        self.epsilon_slider = ttk.Scale(control_frame, from_=0.01, to=2.0, variable=self.epsilon_var, orient=tk.HORIZONTAL)
        self.epsilon_slider.grid(row=0, column=1, sticky="ew", padx=5, pady=2)

        # Slider 2: Carrier Frequency (Hz)
        ttk.Label(control_frame, text="Carrier Freq (Hz):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.freq_var = tk.DoubleVar(value=440.0)
        self.freq_slider = ttk.Scale(control_frame, from_=100, to=2000, variable=self.freq_var, orient=tk.HORIZONTAL)
        self.freq_slider.grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        # Slider 3: Signal Duration (s)
        ttk.Label(control_frame, text="Duration (s):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.duration_var = tk.DoubleVar(value=1.5)
        self.duration_slider = ttk.Scale(control_frame, from_=0.1, to=5.0, variable=self.duration_var, orient=tk.HORIZONTAL)
        self.duration_slider.grid(row=2, column=1, sticky="ew", padx=5, pady=2)

        # Slider 4: Noise Level
        ttk.Label(control_frame, text="Noise Level:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.noise_var = tk.DoubleVar(value=0.05)
        self.noise_slider = ttk.Scale(control_frame, from_=0.0, to=0.5, variable=self.noise_var, orient=tk.HORIZONTAL)
        self.noise_slider.grid(row=3, column=1, sticky="ew", padx=5, pady=2)

        control_frame.columnconfigure(1, weight=1)

        # --- Action Buttons Frame ---
        action_frame = ttk.Frame(self.root, padding=5)
        action_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.trigger_btn = ttk.Button(action_frame, text="⚡ Trigger Simulation", command=self.on_trigger)
        self.trigger_btn.pack(side=tk.LEFT, padx=5)

        self.play_btn = ttk.Button(action_frame, text="▶ Play Audio", command=self.play_audio, state="disabled")
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(action_frame, text="💾 Save Audio (.wav)", command=self.save_audio, state="disabled")
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(action_frame, text="Ready", font=("Arial", 10, "italic"))
        self.status_label.pack(side=tk.LEFT, padx=15)

        # --- Plot Display ---
        plot_frame = ttk.Frame(self.root, padding=10)
        plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Meteor Epsilon Echo Signature")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_trigger(self):
        """Trigger button callback - collects parameters and launches background computation."""
        # Read parameters from UI controls on Main Thread
        params = {
            "epsilon": self.epsilon_var.get(),
            "freq": self.freq_var.get(),
            "duration": self.duration_var.get(),
            "noise": self.noise_var.get(),
        }

        self.trigger_btn.config(state="disabled")
        self.play_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.status_label.config(text="Computing Echo Signature...")

        # Spawn non-blocking thread for math
        threading.Thread(
            target=self._compute_simulation,
            args=(params, self.queue),
            daemon=True
        ).start()

    @staticmethod
    def _compute_simulation(params, q):
        """Worker thread executing background NumPy calculations."""
        try:
            sr = 44100
            t = np.linspace(0, params["duration"], int(sr * params["duration"]), endpoint=False)
            
            # Epsilon Echo decay + carrier signal formula
            carrier = np.sin(2 * np.pi * params["freq"] * t)
            decay = np.exp(-params["epsilon"] * t * 3.0)
            noise = np.random.normal(0, params["noise"], size=t.shape)
            
            signal = (carrier * decay) + noise
            
            # Normalize to 16-bit PCM for audio playback and export
            max_val = np.max(np.abs(signal))
            normalized = signal / max_val if max_val > 0 else signal
            audio_pcm = (normalized * 32767).astype(np.int16)

            q.put(("SUCCESS", (t, signal, audio_pcm)))
        except Exception as e:
            q.put(("ERROR", str(e)))

    def _poll_queue(self):
        """Checks thread queue periodically on the main Tkinter thread."""
        try:
            while True:
                msg_type, payload = self.queue.get_nowait()

                if msg_type == "SUCCESS":
                    t, signal, audio_pcm = payload
                    self.last_audio_data = audio_pcm
                    
                    self._update_plot(t, signal)
                    
                    self.status_label.config(text="Simulation Complete")
                    self.play_btn.config(state="normal")
                    self.save_btn.config(state="normal")

                elif msg_type == "ERROR":
                    self.status_label.config(text=f"Error: {payload}")

                self.trigger_btn.config(state="normal")
        except queue.Empty:
            pass

        self.root.after(50, self._poll_queue)

    def _update_plot(self, t, signal):
        """Update Matplotlib canvas safely on main thread."""
        self.ax.clear()
        self.ax.plot(t, signal, color="navy", linewidth=1, label="Echo Amplitude")
        self.ax.set_title("Meteor Epsilon Echo Signature")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        self.ax.legend(loc="upper right")
        
        self.canvas.draw_idle()

    def play_audio(self):
        """Plays generated signal audio without freezing GUI."""
        if self.last_audio_data is None:
            return

        if HAS_AUDIO:
            play_obj = sa.play_buffer(self.last_audio_data, 1, 2, self.sample_rate)
        else:
            self.status_label.config(text="Install 'simpleaudio' to enable playback!")

    def save_audio(self):
        """Saves current calculation as a .wav file."""
        if self.last_audio_data is None:
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAVE Audio", "*.wav"), ("All Files", "*.*")],
            title="Save Echo Audio"
        )

        if filepath:
            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit PCM (2 bytes)
                wf.setframerate(self.sample_rate)
                wf.writeframes(self.last_audio_data.tobytes())
            
            self.status_label.config(text=f"Saved: {filepath.split('/')[-1]}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MeteorEpsilonEchoSimulator(root)
    root.mainloop()