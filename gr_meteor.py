#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Meteor Scatter Simulation
# Author: amp
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import blocks
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import gr_meteor_epy_block_0 as epy_block_0  # embedded python block
import numpy as np
import sip



class gr_meteor(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Meteor Scatter Simulation", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Meteor Scatter Simulation")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "gr_meteor")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 1e3

        self.noiseLevel = noiseLevel = 0
        self.fade_rate = fade_rate = 0.5
        self.dShift = dShift = 0
        self.burst_duration = burst_duration = 3.0
        self.beacon_freq = beacon_freq = 49.97e6

        ##################################################
        # Blocks
        ##################################################

        self._noiseLevel_range = qtgui.Range(0, 1, 0.05, 0, 200)
        self._noiseLevel_win = qtgui.RangeWidget(self._noiseLevel_range, self.set_noiseLevel, "Noise Level", "slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._noiseLevel_win, 0, 2, 1, 6)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 8):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_sink_x_0 = qtgui.sink_c(
            2048, #fftsize
            window.WIN_BLACKMAN_hARRIS, #wintype
            beacon_freq, #fc
            samp_rate, #bw
            "", #name
            False, #plotfreq
            True, #plotwaterfall
            False, #plottime
            False, #plotconst
            None # parent
        )
        self.qtgui_sink_x_0.set_update_time(1.0/.666)
        self._qtgui_sink_x_0_win = sip.wrapinstance(self.qtgui_sink_x_0.qwidget(), Qt.QWidget)

        self.qtgui_sink_x_0.enable_rf_freq(True)

        self.top_grid_layout.addWidget(self._qtgui_sink_x_0_win, 1, 0, 5, 4)
        for r in range(1, 6):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        if "real" == "int":
        	isFloat = False
        	scaleFactor = 1
        else:
        	isFloat = True
        	scaleFactor = 1

        _qtgui_levelgauge_0_lg_win = qtgui.GrLevelGauge('Doppler',"default","default","default",(-10),(-10), 100, False,1,isFloat,scaleFactor,True,self)
        _qtgui_levelgauge_0_lg_win.setValue(dShift)
        self.qtgui_levelgauge_0 = _qtgui_levelgauge_0_lg_win

        self.top_grid_layout.addWidget(_qtgui_levelgauge_0_lg_win, 0, 0, 1, 2)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.freq_xlating_fir_filter_xxx_0_1 = filter.freq_xlating_fir_filter_ccc(1, [1], (-10), samp_rate)
        self.freq_xlating_fir_filter_xxx_0_1.set_block_alias("doppler_shift")
        self.epy_block_0 = epy_block_0.blk()
        self.blocks_vector_source_x_0 = blocks.vector_source_c(([0.0]*100000 + [np.exp(-x/4000.0) for x in range(10000)] + [0.0]*100000), True, 1, [])
        self.blocks_null_source_0_0_0_0 = blocks.null_source(gr.sizeof_gr_complex*1)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_sig_source_x_0 = analog.sig_source_c(samp_rate, analog.GR_SIN_WAVE, beacon_freq, 1, 0, 0)
        self.analog_sig_source_x_0.set_block_alias("Beacon")
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, noiseLevel, 0)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.epy_block_0, 'freq_out'), (self.freq_xlating_fir_filter_xxx_0_1, 'freq'))
        self.msg_connect((self.epy_block_0, 'freq_out'), (self.qtgui_levelgauge_0, 'value'))
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.qtgui_sink_x_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_add_xx_0, 2))
        self.connect((self.blocks_null_source_0_0_0_0, 0), (self.blocks_add_xx_0, 3))
        self.connect((self.blocks_vector_source_x_0, 0), (self.freq_xlating_fir_filter_xxx_0_1, 0))
        self.connect((self.freq_xlating_fir_filter_xxx_0_1, 0), (self.blocks_multiply_xx_0, 1))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "gr_meteor")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)
        self.qtgui_sink_x_0.set_frequency_range(self.beacon_freq, self.samp_rate)

    def get_qtgui_levelgauge_0(self):
        return self.qtgui_levelgauge_0

    def set_qtgui_levelgauge_0(self, qtgui_levelgauge_0):
        self.qtgui_levelgauge_0 = qtgui_levelgauge_0
        self.qtgui_levelgauge_0.setValue(self.dShift)

    def get_noiseLevel(self):
        return self.noiseLevel

    def set_noiseLevel(self, noiseLevel):
        self.noiseLevel = noiseLevel
        self.analog_noise_source_x_0.set_amplitude(self.noiseLevel)

    def get_fade_rate(self):
        return self.fade_rate

    def set_fade_rate(self, fade_rate):
        self.fade_rate = fade_rate

    def get_dShift(self):
        return self.dShift

    def set_dShift(self, dShift):
        self.dShift = dShift
        self.qtgui_levelgauge_0.setValue(self.dShift)

    def get_burst_duration(self):
        return self.burst_duration

    def set_burst_duration(self, burst_duration):
        self.burst_duration = burst_duration

    def get_beacon_freq(self):
        return self.beacon_freq

    def set_beacon_freq(self, beacon_freq):
        self.beacon_freq = beacon_freq
        self.analog_sig_source_x_0.set_frequency(self.beacon_freq)
        self.qtgui_sink_x_0.set_frequency_range(self.beacon_freq, self.samp_rate)




def main(top_block_cls=gr_meteor, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
