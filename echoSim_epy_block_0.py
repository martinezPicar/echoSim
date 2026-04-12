import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """Meteor echo envelope + Doppler modulator"""

    def __init__(self, samp_rate=48000, avg_rate=8.0,
                 tau_min=0.05, tau_max=0.8,
                 doppler_max=400.0, snr_db=15.0):
        gr.sync_block.__init__(
            self,
            name='Meteor Echo Simulator',
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.samp_rate  = samp_rate
        self.avg_rate   = avg_rate        # meteors per hour (background)
        self.tau_min    = tau_min         # shortest decay, seconds
        self.tau_max    = tau_max         # longest decay, seconds
        self.doppler_max = doppler_max    # max Doppler shift, Hz
        self.gain_lin   = 10 ** (snr_db / 20.0)

        # State
        self._env       = 0.0            # current envelope amplitude
        self._phase     = 0.0            # accumulated Doppler phase
        self._dphi      = 0.0            # phase increment per sample
        self._decay     = 0.0            # per-sample decay factor
        self._next_event = self._draw_interval()

    def _draw_interval(self):
        """Exponential inter-arrival time (Poisson process)."""
        rate_per_sec = self.avg_rate / 3600.0
        return int(np.random.exponential(1.0 / rate_per_sec) * self.samp_rate)

    def _new_meteor(self):
        tau      = np.random.uniform(self.tau_min, self.tau_max)
        doppler  = np.random.uniform(-self.doppler_max, self.doppler_max)
        self._decay = np.exp(-1.0 / (tau * self.samp_rate))
        self._dphi  = 2 * np.pi * doppler / self.samp_rate
        self._env   = self.gain_lin      # peak amplitude
        self._phase = np.random.uniform(0, 2 * np.pi)  # random initial phase

    def work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        n   = len(inp)

        for i in range(n):
            # Countdown to next meteor
            self._next_event -= 1
            if self._next_event <= 0:
                self._new_meteor()
                self._next_event = self._draw_interval()

            # Build modulating phasor: envelope × Doppler
            mod = self._env * np.exp(1j * self._phase)
            out[i] = inp[i] * mod

            # Decay the envelope, advance the phase
            self._env   *= self._decay
            self._phase += self._dphi

        return n
