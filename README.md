# echoSim — Overdense Meteor Echo Simulator

A Python-based interactive simulator that synthesizes and visualizes overdense meteor radio echoes
under **forward-scatter geometry**, modeling wind shear deformation of the meteor trail and the
resulting Doppler signatures in a spectrogram.

![GUI](https://img.shields.io/badge/GUI-Tk%2FMatplotlib-darkblue)
![Language](https://img.shields.io/badge/Python-3.8%2B-yellow)
![Domain](https://img.shields.io/badge/Domain-Radio%20Meteor%20Astronomy-blueviolet)
<!--
[![Paper](https://img.shields.io/badge/Paper-IMC%202026%20Barcelonnette-informational)](https://github.com/martinezPicar/echoSim)

> Accompanying code of: A. Martínez Picar (Royal Observatory of Belgium),
> *"Yet another epsilon meteor echo simulator"*, Proceedings of the IMC, Barcelonnette, 2026.
-->
---

## Overview

**echoSim** simulates how a meteor trail — treated as a **forward-scatter reflector** between a
transmitter (Tx) and a receiver (Rx) — evolves over time under horizontal wind shear. The trail
deforms spatially, its specular reflection zone narrows or broadens, and the resulting
echo produces characteristic **Doppler-shifted tones** that vary in time, reproducing the classic
**C- and $\epsilon$-shaped** signatures of exotic overdense radio meteor echoes and their relation to
atmospheric dynamics.

The simulation is embedded in an interactive desktop GUI with:

- **10 live-tunable parameters** (sliders)
- A **2D spatial plot** of trail deformation over time
- A real-time **spectrogram** of the synthesized echo signal
- Audio **playback** and export to **WAV** / **PNG**

The goal is not to reproduce every aspect of meteor-plasma physics, but to provide a
**controllable numerical experiment** in which the main geometrical and kinematic ingredients
can be varied independently, producing outputs in good *qualitative* agreement with
observational data.

---

## Physics Model

### 1. Forward-Scatter Geometry

A generalized bistatic Tx–Rx geometry is constructed by `build_forward_scatter_geometry()`,
parameterized by baseline distance $d$, specular altitude $h_{\mathrm{spec}}$, meteor azimuth
$\alpha$ and elevation $\eta$. Tx and Rx sit symmetrically at $\mp d/2$ relative to a reference
point $p_{\mathrm{ref}} = (0, 0, h_{\mathrm{spec}})$.

The horizontal meteor direction and the 3-D trajectory direction are

$$\hat{\mathbf{x}} = \sin\alpha\,\hat{\mathbf{e}} + \cos\alpha\,\hat{\mathbf{n}},
\qquad
\hat{\mathbf{t}} = \cos\eta\,\hat{\mathbf{x}} + \sin\eta\,\hat{\mathbf{z}}$$

where $\hat{\mathbf{e}}$, $\hat{\mathbf{n}}$, $\hat{\mathbf{z}}$ are the east, north and up unit
vectors. The trail is uniformly sampled at **4000 points** over $z \in [-1.2, +1.2]$ km, and each
trail point is placed at

$$\mathbf{p}(z) = \mathbf{r}_{\mathrm{ref}} + x_0(z)\hat{\mathbf{x}} + z\hat{\mathbf{z}}$$

where $x_0(z) = z \tan\sigma$ is the initial horizontal offset produced by the **trail skew
angle** $\sigma$ (initial tilt of the ionized trail relative to the local vertical, in the plane
perpendicular to the trajectory), so that $dx_0/dz = \tan\sigma$.

For each trail point $p_i$:

- unit vectors $\hat{u}_T$, $\hat{u}_R$ point from Tx and Rx to the point;
- the **bistatic angle** is $\beta_i = \arccos(\hat{u}_T \cdot \hat{u}_R)$;
- the **bistatic bisector** is $\hat{b}_i = (\hat{u}_T + \hat{u}_R)/\lVert \hat{u}_T + \hat{u}_R \rVert$;
- the **specular point** is the point minimizing the total path length $r_T + r_R$.

### 2. Wind Shear Field

Horizontal wind velocity along the trail is decomposed into three harmonics:

$$v(z) = v_{\mathrm{wind}}\left[a_1 \sin(\pi z) + a_2 \sin(2\pi z) + a_3 \sin(3\pi z)\right]$$

with $z$ in km, $v_{\mathrm{wind}}$ the overall wind-speed scale (m/s), and $a_1, a_2, a_3$ the
relative amplitudes of the fundamental, second and third harmonics. The derivative $dv/dz$
drives the shear deformation of the trail over time.

### 3. Signal Synthesis

`synthesize_meteor_signal()` builds a 60-second timeline; the meteor is active only for a finite
**15 s** interval beginning at $t_0 = 20$ s.

#### Envelope

The echo amplitude is modulated by a global envelope $E(t)$ representing trail formation and
delayed diffusion:

- zero before $t_0$ and after $t_0 + T_{\mathrm{active}}$;
- from $t_0$ to $t_0 + t_{\mathrm{diff}}$ it rises as $1 - \exp\left[-(t-t_0)/\tau_{\mathrm{rise}}\right]$;
- after the diffusion delay, this rising term is multiplied by
  $\exp\left[-(t - t_0 - t_{\mathrm{diff}})/\tau_{\mathrm{decay}}\right]$ until the end of the active interval.

Implementation constants: $\tau_{\mathrm{rise}} = 5$ ms, $t_{\mathrm{diff}} = 1$ s,
$\tau_{\mathrm{decay}} = 1$ s, $T_{\mathrm{active}} = 15$ s.

| Phase | Time window | Behavior |
|-------|-------------|----------|
| Background | t < 20&nbsp;s | Pure background noise (if enabled) |
| Meteor entry | t = 20~s ($\tau$&nbsp;=&nbsp;0) | Exponential rise; fast decay of the initial "ping" |
| Active echo | 20&nbsp;s ≤ t < 35&nbsp;s | Diffusion-limited decay; wind shear progressively deforms the trail |
| Background | t ≥ 35&nbsp;s | Pure background noise (if enabled) |

#### Ping vs. shear blending

At each time $\tau$ after trail formation, the horizontal velocity field is formed from a decaying
entry component and a growing wind-shear component:

$$v_{\mathrm{hor}}(\tau) = e^{-\tau / \tau_{\mathrm{ping}}} v_{\mathrm{entry}} + \left(1 - e^{-\tau / \tau_{\mathrm{shear}}}\right) v(z)$$

with $\tau_{\mathrm{ping}} = 0.12$ s and $\tau_{\mathrm{shear}} = 0.40$ s. The entry velocity is

$$v_{\mathrm{entry}}(z) = 8 \sin(2\pi z) + 15 \tan(\sigma) \quad \mathrm{(m/s)}$$

The two time-dependent coefficients are applied independently and do not form a normalized interpolation.

#### Doppler shift

The 3-D velocity at each trail point is

$$\mathbf{v}_i = v_{\mathrm{hor},i} \bigl(\cos\eta \hat{\mathbf{x}} + \sin\eta \hat{\mathbf{z}}\bigr)$$

and the instantaneous Doppler shift is the projection onto the bistatic bisector, normalized by
the wavelength $\lambda = c/f_0$:

$$f_{\mathrm{D},i} = \frac{\mathbf{v}_i \cdot \hat{\mathbf{b}}_i}{\lambda}$$

#### Specularity weighting

The contribution of each trail segment is weighted by a Gaussian specularity factor

$$w_i(\tau) = \exp \left[-\frac{(dx/dz)_i^2}{A(\tau)}\right]$$

with instantaneous trail slope

$$\left(\frac{dx}{dz}\right)_i = \left(\frac{dx_0}{dz}\right)_i + \left(\frac{dv}{dz}\right)_i \tau \left(1 - e^{-\tau/\tau_{\mathrm{shear}}}\right) 0.0025$$

(the factor 0.0025 is an empirical unit-conversion scaling for the $(\mathrm{m/s})/\mathrm{km}$
shear rate), and time-dependent aperture

$$A(\tau) = 0.005 + 0.015\left(1 - e^{-\tau/0.40}\right) + 0.010\,\max(0, \tau - 1)$$

Thus the aperture increases with the shear timescale and, after the 1 s diffusion delay,
increases linearly with elapsed time.

#### Phase accumulation and noise

The composite audio signal is synthesized by coherent phase accumulation:

$$\phi_i(t + \Delta t) = \phi_i(t) + 2\pi\,\bigl[f_c + f_{\mathrm{D},i}(t)\bigr]\,\Delta t$$
$$s(t) = E(t)\sum_i w_i(t)\,\sin\phi_i(t) + n(t)$$

where $f_c = 1000$ Hz is the audio carrier and $n(t)$ is optional zero-mean Gaussian noise. When
enabled, the code estimates the signal power over the active interval, sets the noise standard
deviation from the user-defined SNR (**18 dB** by default), adds the noise to the complete 60 s
waveform, and normalizes the final waveform.

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
| **Tx-Rx Distance** $d$ | 20 – 200 km | 50 km | Ground baseline between Tx and Rx |
| **Specular Alt** $h_{\mathrm{spec}}$ | 85 – 130 km | 90 km | Altitude of the meteor specular point |
| **Carrier Frequency** $f_0$ | 30 – 300 MHz | 50 MHz | Radar carrier frequency |
| **Trail Skew** $\sigma$ | 1 – 45° | 12° | Initial tilt of the trail from vertical |
| **Wind Speed** $v_{\mathrm{wind}}$ | 20 – 150 m/s | 80 m/s | Amplitude of the wind shear field |
| **Fundamental Ratio** $a_1$ | 0 – 1 | 1.00 | Weight of $\sin(\pi z)$ wind component |
| **2nd Harmonic Ratio** $a_2$ | 0 – 1 | 0.00 | Weight of $\sin(2\pi z)$ wind component |
| **3rd Harmonic Ratio** $a_3$ | 0 – 1 | 0.10 | Weight of $\sin(3\pi z)$ wind component |
| **Azimuth ($\alpha$)** | 0 – 360° | 180° | Meteor trajectory azimuth |
| **Elevation ($\eta$)** | 0 – 90° | 45° | Meteor trajectory elevation |

### Buttons & Toggles

| Control | Action |
|---------|--------|
| **Add Background Noise** | Toggle Gaussian noise at 18 dB SNR |
| **Run Simulation** | Re-synthesize signal and refresh both plots |
| **Play Audio** | Play the synthesized echo through the default audio device |
| **Save Audio File (.wav)** | Export 16-bit PCM WAV via file dialog |
| **Save Spectrogram (.png)** | Export the spectrogram axes at 600 DPI via file dialog |

---

## Display Panels

### Left — Trail Deformation

Plots horizontal position $x$ vs. relative altitude $z$ for the initial trail (dashed white)
and four deformed snapshots ($t = 1.5, 3.0, 4.5, 6.0$~s) colored by the spring colormap.
The title reports the current geometry, wind speed (m/s) and the specular bistatic angle $\beta$.

### Right — Spectrogram

Log-magnitude spectrogram of the synthesized audio, computed with `NFFT = 16384` and
`noverlap = 14254` (~79% overlap), windowed to ±120 Hz around the 1 kHz carrier and displayed
over the full 60 s timeline. The meteor event appears between 20 s and 35 s as a structured,
time-varying Doppler signature.

---

## Typical Output Features

For a representative parameter set ($d = 50$ km, $h_{\mathrm{spec}} = 90$ km, $f_0 = 50$ MHz,
$\alpha = 180^{\circ}$, $\eta = 75^{\circ}$, $v_{\mathrm{wind}} = 80$ m/s), the simulated
spectrogram shows:

- a **rapid initial "ping"** at $t \approx 20$ s from the initial specular reflection off the
  newly formed trail;
- a **smooth transition to Doppler spreading** induced by wind shear over 2–3 s;
- a **gradual decrease in echo amplitude** after the diffusion delay, while the evolving
  specularity weighting modifies the distribution of Doppler contributions.

These features reproduce, qualitatively, the observed C- and ε-shaped overdense meteor echoes
and multi-branch events recorded by forward-scatter stations (see the paper, Figs. 4–6).
The comparisons are qualitative; the simulator is not intended to reproduce the complete
physical plasma dynamics of observed events.

---

## Code Structure

```
echoSim.py
├── Constants & geometry
│   └── C_LIGHT
├── build_forward_scatter_geometry(...)   # bistatic geometry, specular point, bisector
├── synthesize_meteor_signal(...)         # full signal synthesis pipeline
│   ├── timing & delayed-diffusion envelope
│   ├── altitude grid, skew & multi-harmonic wind field
│   ├── per-sample ping/shear blending, Doppler, specularity
│   ├── coherent phase accumulation
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
| `t_entry` | 20 s | Meteor entry time $t_0$ |
| `meteor_duration` | 15 s | Active echo duration $T_{\mathrm{active}}$ |
| `t_diffusion_start` | 1 s | Delay $t_{\mathrm{diff}}$ before diffusion decay begins |
| `tau_ping_decay` | 0.12 s | Initial ping decay $\tau_{\mathrm{ping}}$ |
| `tau_shear_growth` | 0.40 s | Wind-shear growth $\tau_{\mathrm{shear}}$ |
| `CARRIER_FREQ` | 1000 Hz | Audio carrier $f_c$ |
| `snr_db` | 18 dB | Signal-to-noise ratio |
| `SAMPLE_RATE` | 22050 Hz | Audio sample rate (22.05 kHz) |

---

## Example Signatures to Explore

| Setting | Expected behavior |
|---------|-------------------|
| Increase **3rd Harmonic Ratio** $a_3$ | More complex, multi-branched Doppler curves |
| Large **Trail Skew** $\sigma$ | Weaker specular return; broader aperture needed |
| Change **Azimuth** $\alpha$ | Alters the projection of wind velocity onto the bistatic bisector → Doppler scale |
| Increase **Wind Speed** $v_{\mathrm{wind}}$ | Faster trail deformation; broader Doppler spread over time |
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
 <!--
## Citation

If you use echoSim in your work, please cite:

> Martínez Picar, A. (2026). *Yet another epsilon meteor echo simulator.*
> Proceedings of the International Meteor Conference (IMC), Barcelonnette, 2026.
> Royal Observatory of Belgium, Brussels.

**Source code:** https://github.com/martinezPicar/echoSim
-->
## Acknowledgements

The author gratefully acknowledges that a significant part of the development of this
simulation code was inspired by the earlier work of **Peter Martinez (G3PLX)** and by the
images of his "Meteor Ping Simulator"
[published online](https://g3zjoradio.wordpress.com/2017/11/25/in-persuit-of-c-shape-meteor-reflections/)
by **Eddie Bennett (G3ZJO)**.

<!--
## References

- Belkovich, O. I. (2006). "The physics of backscattering of radio waves from ionized meteor
  trails." In Verbeeck C. and Wislez J.-M. (eds.), *Proceedings of the Radio Meteor School 2005*, pp. 27–33.
- Rault, J.-L. (2010). "Epsilons, c's, corkscrews, and co." Presentation at the International
  Meteor Conference 2010, Armagh, Northern Ireland, UK.
- Steyaert, C. (2012). "Epsilons: we need more theories." In *Proceedings of the International
  Meteor Conference*, 30th IMC, Sibiu, Romania, 2011, pp. 64–68.
-->