# Speaker Notes: "Yet Another Epsilon Meteor Echo Simulator"
## IMC 2026 — 15-minute talk (including ~3 min questions)

---

## Timing Overview

| Section | Slide(s) | Time | Cumulative |
|---------|----------|------|------------|
| Title & intro | 1 | 0:30 | 0:30 |
| Motivation | 2–3 | 1:30 | 2:00 |
| Geometry | 4–5 | 2:00 | 4:00 |
| Physical model | 6–9 | 4:00 | 8:00 |
| Implementation | 10–11 | 2:00 | 10:00 |
| Results | 12–13 | 2:00 | 12:00 |
| Conclusion | 14–15 | 1:30 | 13:30 |
| Questions | — | 1:30 | 15:00 |

---

## Slide-by-Slide Notes

### Slide 1: Title
- Welcome audience, introduce yourself
- Mention this is an open-source tool for education and research

### Slide 2: Why Yet Another Simulator?
- Emphasize the "gap" in existing tools
- Analytical models = fast but only underdense
- Full-wave = accurate but slow
- We need something in between for intuition-building

### Slide 3: Outline
- Brief walkthrough of the three main sections

### Slide 4: Bistatic Geometry Setup
- Point to the diagram: Tx/Rx on ground, specular point at 90 km
- Explain that azimuth and elevation fully determine the trajectory
- The bisector direction is key for Doppler sensitivity

### Slide 5: Specular Point & Bistatic Angle
- Highlight that the specular point is found by minimizing path length
- The bistatic angle beta determines the scattering geometry
- Mention that this is all computed analytically — no lookup tables

### Slide 6: Temporal Envelope
- Walk through the equation: formation (rise) and diffusion (decay)
- The 5 ms rise time captures the initial ionization burst
- The 3-second diffusion onset is characteristic of overdense trails at 90 km

### Slide 7: Wind-Shear Field
- Explain the harmonic expansion — this is the "secret sauce"
- Most simulators use linear wind; we use multi-harmonic to match real profiles
- Mention that users can set r2, r3 to zero for simple linear shear

### Slide 8: Velocity Evolution
- This is the most important physical slide
- The echo starts as a head echo (ping) and transitions to wind-driven shear
- The time constants (0.12 s and 0.40 s) were chosen to match typical overdense echo durations

### Slide 9: Doppler & Specular Weighting
- Doppler comes from projecting velocity onto the bisector
- The specular weight is like a Fresnel zone that gets broader as the trail twists
- This explains why echoes narrow in bandwidth before fading

### Slide 10: Signal Synthesis Pipeline
- Mention 4000 points along the trail, 15 seconds at 22.05 kHz
- The phase accumulation approach avoids FFT artifacts
- Runtime is ~0.5 s — fast enough for interactive use

### Slide 11: Interactive GUI
- If possible, show a short screen recording or live demo (30 seconds)
- Highlight the two-panel layout: geometry left, spectrogram right
- Mention the 10 sliders cover the full parameter space

### Slide 12: Default Parameter Echo
- Describe the characteristic morphology: ping → drift → decay
- This matches what radio observers typically record

### Slide 13: Parameter Space Exploration
- Give 2–3 concrete examples of how changing parameters affects the echo
- High r2 → S-curve (hook echo)
- High wind → large Doppler excursion
- Large baseline → weaker, longer echo

### Slide 14: Summary & Future Work
- Recap the three key features: geometry, wind shear, interactivity
- Future work: underdense regime, empirical wind models, multi-station

### Slide 15: Thank You / Questions
- Open the floor for questions
- Be ready to discuss: (1) why 22.05 kHz audio, (2) validation against real data, (3) computational performance

---

## Anticipated Questions & Suggested Answers

**Q: Why is the audio carrier 1 kHz?**
A: It is an arbitrary choice for audible playback. The actual Doppler shifts are scaled from the RF carrier (e.g., 50 MHz) to audio frequencies by preserving the fractional shift. This makes the simulation audible for educational demos.

**Q: Have you validated this against real meteor echoes?**
A: Qualitative comparison shows good agreement with typical overdense echo morphologies. Quantitative validation against calibrated radar data is ongoing and will be the subject of a future paper.

**Q: Why Python and not C++/Fortran?**
A: Python with NumPy is fast enough for this level of synthesis (~0.5 s). The priority was accessibility and hackability. Bottlenecks could be JIT-compiled with Numba if needed.

**Q: Can this simulate underdense echoes?**
A: Not yet — the current model uses specular weighting appropriate for overdense trails. Fresnel-zone diffraction for underdense echoes is planned as a future extension.

**Q: What does "epsilon" in the title refer to?**
A: It is partly a playful reference to the "Yet Another..." programming trope, and partly a nod to the small parameter $\epsilon$ in perturbation theory — our model is a minimal but physically meaningful approximation.

---

## Technical Tips for Presenting

- If compiling the Beamer slides, use `pdflatex` or `lualatex`
- The aspect ratio is 16:9 (`aspectratio=169`)
- Consider replacing placeholder images with actual screenshots from `echoSim.py`
- For a live demo, have a pre-recorded video as backup
- Bring business cards or QR codes linking to the repository
