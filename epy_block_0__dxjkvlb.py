import random
from gnuradio import gr
import pmt

class blk(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name="RandomFreqUpdater",
            in_sig=None,
            out_sig=None
        )

        # Add a message output port
        self.message_port_register_out(pmt.intern("freq_out"))

        self.counter = 0
        self.current_freq = 0

    def work(self, input_items, output_items):
        self.counter += 1

        if self.counter % 100 == 0:   # update periodically
            self.current_freq = random.uniform(-10, 10)
            self.set_var("dShift", self.current_freq)

            # publish as PMT float
            self.message_port_pub(
                pmt.intern("freq_out"),
                pmt.from_double(self.current_freq)
            )

        return 0

