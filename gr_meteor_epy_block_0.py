import random
import numpy as np
from gnuradio import gr
import pmt

class blk(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name="Random dShift Publisher",
            in_sig=None,
            out_sig=[np.float32]   # optional float output stream
        )
        self.counter = 0
        self.current_val = 0.0

        # Message output port for variable updates
        self.message_port_register_out(pmt.intern("freq_out"))

    def work(self, input_items, output_items):
        out = output_items[0]

        for i in range(len(out)):
            if self.counter % 12096 == 0:
                self.current_val = random.uniform(-25, 25)

                # Build PMT pair: (variable name, value)
                msg = pmt.cons(
                    pmt.intern("dShift"),
                    pmt.from_double(self.current_val)
                )

                # Publish the message
                self.message_port_pub(pmt.intern("freq_out"), msg)

                # Print to terminal
                # print(f"[Random dShift Publisher] dShift = {self.current_val:.3f}")

            out[i] = self.current_val
            self.counter += 1

        return len(out)

