# echoSim — Overdense Meteor Echo Simulator

A Python-based interactive simulator that synthesizes and visualizes overdense meteor radio echoes
under **forward-scatter geometry**, modeling wind shear deformation of the meteor trail and the
resulting Doppler signatures in a spectrogram.

![GUI](https://img.shields.io/badge/GUI-Tk%2FMatplotlib-darkblue)
![Language](https://img.shields.io/badge/Python-3.8%2B-yellow)
![Domain](https://img.shields.io/badge/Domain-Radio%20Meteor%20Astronomy-blueviolet)

---

## Overview

**echoSim** simulates how a meteor trail — treated as a **forward-scatter reflector** between a
transmitter (Tx) and a receiver (Rx) — evolves over time under horizontal wind shear. The trail
deforms spatially, its specular reflection zone narrows or broadens, and the resulting
echo produces characteristic **Doppler-shifted tones** that vary in time, producing the classic
"whistling" meteor echo signature on a spectrogram.

The simulation is embedded in an interactive desktop GUI with:

- **10 live-tunable parameters** (sliders)
- A **2D spatial plot** of trail deformation over time
- A real-time **spectrogram** of the synthesized echo signal
- Audio **playback** and export to **WAV** / **PNG**

---

## Physics Model

### 1. Forward-Scatter Geometry

A generalized bistatic geometry is constructed by `build_forward_scatter_geometry()`:

- Transmitter and receiver separated by distance $d$ on the ground baseline
- Meteor trajectory defined by **azimuth** $\alpha$ and **elevation** $\eta$, referenced to a specular
  point at altitude $h_{\mathrm{spec}}$
- For each point $P_i$ along the trail, the total path length $r_T + r_R$ (Tx→P→Rx) is computed;
  the point of minimum path defines the **specular reflection region**
- The **bistatic bisector** (of the angle $\beta$ between Tx→P and P→Rx directions) sets the
  effective reflection axis used for Doppler computation
- Doppler shifts are computed as $f_{\mathrm{D}} = \frac{v \cdot \hat{b}}{\lambda}$, where $\lambda=c/f_0$

### 2. Wind Shear Field

Horizontal wind velocity along the trail altitude coordinate `z` is modeled as a
**multi-harmonic sine series**:

$v(z) = v_{\mathrm{wind}} [a_1 \sin(\pi z) + a_2 \sin(2 \pi z) + a_3 \sin (3 \pi z)]$

- $a_1$ — fundamental (main shear) ratio
- $a_2$ — second harmonic ratio
- $a_3$ — third harmonic ratio

This produces realistic, vertically varying wind profiles rather than a uniform drift.

### 3. Signal Synthesis

`synthesize_meteor_signal()` builds a 60-second timeline:

| Phase | Time window | Behavior |
|-------|-------------|----------|
| Silence | t < 20 s | Pure background noise (if enabled) |
| Meteor entry | t = 20 s ($\tau = 0$) | Exponential rise envelope, fast decay of the initial "ping" |
| Active echo | 20 s ≤ t < 35 s | Diffusion-limited decay; wind shear progressively deforms the trail |
| Silence | t ≥ 35 s | Pure background noise (if enabled) |

Key mechanisms:

- **Envelope** — fast rise ($\tau_{\mathrm{rise}} \approx5~\mathrm{ms}$), diffusion decay starting 1 s after entry
- **Ping vs. shear blending** — initial specular response (meteor head/entry velocity
  profile $v_{\mathrm{entry}}$) decays with $\tau = 0.12~\mathrm{s}$, while the wind-shear-driven trail echo
  grows in with $\tau \approx 0.40~\mathrm{s}$
- **Trail deformation** — the initial trail tilt (*skew* angle $\sigma$) accumulates additional
  $dv/dz$ deformation over time
- **Specular weighting** — a Gaussian aperture in $dx/dz$ selects near-perpendicular
  reflection points; the aperture **broadens with time** (diffusion)
- **Carrier** — the echo is synthesized around a 1&nbsp;kHz audio carrier representing the
  radar carrier frequency; SNR-controlled Gaussian noise can be added

---

## Installation

### Requirements

```bash
pip install numpy scipy matplotlib
```

### Optional (audio playback)

```bash
pip install sounddevice
```

> Without `sounddevice`, everything works except real-time audio playback.

---

## Usage

```bash
python echoSim.py
```

The GUI window opens with default parameters. Click **Run Simulation** after adjusting
sliders, then **Play Audio**, **Save Audio File (.wav)**, or **Save Spectrogram (.png)**.

---

## GUI Controls

### Sliders

| Control | Range | Default | Description |
|---------|-------|---------|-------------|
| **Tx-Rx Distance** | 20 – 200 km | 50 km | Ground baseline between Tx and Rx |
| **Specular Alt** | 85 – 130 km | 90 km | Altitude of the meteor specular point |
| **Carrier Frequency** | 30 – 300 MHz | 50 MHz | Radar carrier frequency f₀ |
| **Trail Skew** | 1 – 45° | 12° | Initial tilt of the trail from vertical |
| **Wind Speed** | 20 – 150 m/s | 80 m/s | Amplitude of the wind shear field |
| **Fundamental Ratio** | 0 – 1 | 1.00 | Weight of sin(πz) wind component |
| **2nd Harmonic Ratio** | 0 – 1 | 0.00 | Weight of sin(2πz) wind component |
| **3rd Harmonic Ratio** | 0 – 1 | 0.10 | Weight of sin(3πz) wind component |
| **Azimuth ($\alpha$)** | 0 – 360° | 180° | Meteor trajectory azimuth |
| **Elevation ($\eta$)** | 0 – 90° | 45° | Meteor trajectory elevation |

### Buttons & Toggles

| Control | Action |
|---------|--------|
| ☑ **Add Background Noise** | Toggle Gaussian noise at 18 dB SNR |
| ▶ **Run Simulation** | Re-synthesize signal and refresh both plots |
| 🔊 **Play Audio** | Play the synthesized echo through the default audio device |
| 💾 **Save Audio File (.wav)** | Export 16-bit PCM WAV via file dialog |
| 🖼 **Save Spectrogram (.png)** | Export the spectrogram axes at 600 DPI via file dialog |

---

## Display Panels

### Left — Trail Deformation

Plots horizontal position $x$ vs. relative altitude $z$ for the initial trail (dashed white)
and four deformed snapshots $(t = 1.5, 3.0, 4.5, 6.0~\mathrm{s})$ colored by the spring colormap.
Title reports current geometry and the specular bistatic angle $\beta$.

### Right — Spectrogram

Log-magnitude spectrogram (`NFFT = 16384`, ~79% overlap) of the synthesized audio,
windowed to ±120 Hz around the 1 kHz carrier. The meteor event appears between 20&nbsp;s and
35&nbsp;s as a structured, time-varying Doppler signature. Axis range: full 60&nbsp;s timeline.

---

## Code Structure

```
echoSim.py
├── Constants & geometry
│   └── C_LIGHT
├── build_forward_scatter_geometry(...)   # bistatic geometry, specular point, bisector
├── synthesize_meteor_signal(...)         # full signal synthesis pipeline
│   ├── timing & envelope
│   ├── altitude grid & wind field
│   ├── per-sample Doppler + reflection loop
│   └── noise injection & normalization
└── main()                                # GUI construction & event wiring
    ├── sliders / buttons / checkboxes
    ├── update_plots()                    # re-synthesize + redraw
    ├── play_audio_event()
    ├── save_audio_event()
    └── save_spectrogram_event()
```

### Key constants (tweakable in source)

| Symbol | Value | Meaning |
|--------|-------|---------|
| `t_entry` | 20 s | Meteor entry time |
| `meteor_duration` | 15 s | Active echo duration |
| `t_diffusion_start` | 1 s | Delay before diffusion decay begins |
| `tau_ping_decay` | 0.12 s | Initial ping decay constant |
| `tau_shear_growth` | 0.40 s | Wind-shear echo growth constant |
| `CARRIER_FREQ` | 1000 Hz | Audio carrier representing f₀ |
| `snr_db` | 18 dB | Signal-to-noise ratio |
| `SAMPLE_RATE` | 22050 Hz | Audio sample rate |

---

## Example Signatures to Explore

| Setting | Expected behavior |
|---------|-------------------|
| Increase **3rd Harmonic Ratio** | More complex, multi-branched Doppler curves |
| Large **Trail Skew** | Weaker specular return; broader aperture needed |
| Change **Azimuth** | Alters the projection of wind velocity onto the bistatic bisector → Doppler scale |
| Increase **Wind Speed** | Faster trail deformation; broader Doppler spread over time |
| Toggle noise off | Clean spectrogram of the pure echo signal |

---

## Notes & Limitations

- The `sounddevice` import is optional; the script degrades gracefully to a no-playback mode.
- GUI requires a display with the `TkAgg` Matplotlib backend (standard on desktop Linux,
  Windows, macOS).
- The 15 s active window is defined relative to the start of the echo event (t = 20 s);
  editing `t_entry` or `meteor_duration` shifts this window.
- Signal synthesis loops over active samples in Python; longer durations or higher sample
  rates increase computation time linearly.

---

### Acknowledgements

The author gratefully acknowledges that a significant part of the development of the simulation code presented in this paper was inspired by the earlier work of Peter Martinez (G3PLX) and by the images of his “Meteor Ping Simulator” [published online](https://g3zjoradio.wordpress.com/2017/11/25/in-persuit-of-c-shape-meteor-reflections/) by Eddie Bennett (G3ZJO).
