#!/usr/bin/env python3
import numpy as np
from gnuradio import gr, blocks, analog, filter
from gnuradio.fft import window

class MeteorScatterSim(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "Meteor Scatter Simulation")

        # Parameters
        self.samp_rate = 1e6
        self.burst_freq = 0.3
        self.doppler_max = 2000
        self.noise_level = 0.02
        self.burst_duration = 0.3

        # Signal Generation
        self.signal = analog.sig_source_c(
            self.samp_rate, analog.GR_SIN_WAVE, 10e3, 1.0)

        # Meteor Burst Simulation
        self.trigger = analog.sig_source_f(
            self.samp_rate, analog.GR_CONST_WAVE, 
            self.burst_freq, 1.0, 0)
        
        self.thresh = blocks.threshold_ff(0.5, 0.5, 0)
        self.smooth = filter.fir_filter_fff(
            1, np.ones(int(self.samp_rate*self.burst_duration/10)))

        # Doppler Simulation
        self.doppler_ctrl = analog.sig_source_f(
            self.samp_rate, analog.GR_SAW_WAVE,
            self.burst_freq, self.doppler_max, 0)
        
        self.vco = analog.sig_source_c(self.samp_rate, analog.GR_SIN_WAVE, 0, 1.0)
        self.freq_mod = blocks.multiply_cc()

        # Channel Effects
        self.delay = blocks.delay(gr.sizeof_gr_complex, int(0.001*self.samp_rate))
        self.noise = analog.noise_source_c(analog.GR_GAUSSIAN, self.noise_level)
        self.adder = blocks.add_cc()

        # File Sinks instead of QT
        self.file_sink = blocks.file_sink(gr.sizeof_gr_complex, "meteor_output.dat")
        self.file_sink.set_unbuffered(False)

        # Connections
        self.connect(self.trigger, self.thresh, self.smooth)
        self.connect(self.signal, (self.freq_mod, 0))
        self.connect(self.vco, (self.freq_mod, 1))
        self.connect(self.freq_mod, self.delay, (self.adder, 0))
        self.connect(self.noise, (self.adder, 1))
        
        # Convert burst envelope to complex
        self.float_to_complex = blocks.float_to_complex()
        zero_source = analog.sig_source_f(self.samp_rate, analog.GR_CONST_WAVE, 0, 0, 0)
        self.connect(self.smooth, (self.float_to_complex, 0))
        self.connect(zero_source, (self.float_to_complex, 1))
        
        self.burst_gate = blocks.multiply_cc()
        self.connect(self.adder, (self.burst_gate, 0))
        self.connect(self.float_to_complex, (self.burst_gate, 1))
        self.connect(self.burst_gate, self.file_sink)

    def start(self):
        super().start()
        print("Simulation running - processing 10 seconds of data...")
        import time
        time.sleep(10)
        self.stop()
        self.wait()
        print("Simulation complete. Data saved to meteor_output.dat")

if __name__ == "__main__":
    tb = MeteorScatterSim()
    tb.start()
