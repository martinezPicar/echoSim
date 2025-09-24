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
from gnuradio import eng_notation
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
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
        self.ud_dur = ud_dur = 0.5
        self.samp_rate = samp_rate = 6048
        self.ud_spacing = ud_spacing = 10
        self.ud_samples = ud_samples = samp_rate*ud_dur
        self.samp_rate_label = samp_rate_label = samp_rate
        self.riseFracc = riseFracc = 0.1
        self.noiseLevel = noiseLevel = 0.10
        self.beacon_freq = beacon_freq = 49970000
        self.beacon_att = beacon_att = 1

        ##################################################
        # Blocks
        ##################################################

        self._ud_spacing_range = qtgui.Range(1, 120, 1, 10, 200)
        self._ud_spacing_win = qtgui.RangeWidget(self._ud_spacing_range, self.set_ud_spacing, "Underdense Time Spacing", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._ud_spacing_win)
        self._riseFracc_range = qtgui.Range(0.05, 0.5, 0.05, 0.1, 200)
        self._riseFracc_win = qtgui.RangeWidget(self._riseFracc_range, self.set_riseFracc, "Fraction of Rising Power (Profile)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._riseFracc_win)
        self._noiseLevel_range = qtgui.Range(0, 1, 0.05, 0.10, 200)
        self._noiseLevel_win = qtgui.RangeWidget(self._noiseLevel_range, self.set_noiseLevel, "Noise Level", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._noiseLevel_win)
        self.display = Qt.QTabWidget()
        self.display_widget_0 = Qt.QWidget()
        self.display_layout_0 = Qt.QBoxLayout(Qt.QBoxLayout.TopToBottom, self.display_widget_0)
        self.display_grid_layout_0 = Qt.QGridLayout()
        self.display_layout_0.addLayout(self.display_grid_layout_0)
        self.display.addTab(self.display_widget_0, 'Displays')
        self.top_layout.addWidget(self.display)
        self._beacon_freq_tool_bar = Qt.QToolBar(self)
        self._beacon_freq_tool_bar.addWidget(Qt.QLabel("Beacon Frequency" + ": "))
        self._beacon_freq_line_edit = Qt.QLineEdit(str(self.beacon_freq))
        self._beacon_freq_tool_bar.addWidget(self._beacon_freq_line_edit)
        self._beacon_freq_line_edit.editingFinished.connect(
            lambda: self.set_beacon_freq(eng_notation.str_to_num(str(self._beacon_freq_line_edit.text()))))
        self.top_layout.addWidget(self._beacon_freq_tool_bar)
        self._beacon_att_range = qtgui.Range(0, 1, 0.05, 1, 200)
        self._beacon_att_win = qtgui.RangeWidget(self._beacon_att_range, self.set_beacon_att, "Direct Beacon 'Gain'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._beacon_att_win)
        self._ud_dur_range = qtgui.Range(0, 1, 0.1, 0.5, 200)
        self._ud_dur_win = qtgui.RangeWidget(self._ud_dur_range, self.set_ud_dur, "Underdense Total Duration", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._ud_dur_win)
        self._samp_rate_label_tool_bar = Qt.QToolBar(self)

        if None:
            self._samp_rate_label_formatter = None
        else:
            self._samp_rate_label_formatter = lambda x: str(x)

        self._samp_rate_label_tool_bar.addWidget(Qt.QLabel("Sampling Rate (Sa/s):"))
        self._samp_rate_label_label = Qt.QLabel(str(self._samp_rate_label_formatter(self.samp_rate_label)))
        self._samp_rate_label_tool_bar.addWidget(self._samp_rate_label_label)
        self.top_layout.addWidget(self._samp_rate_label_tool_bar)
        self.qtgui_time_sink_x_0 = qtgui.time_sink_c(
            1024, #size
            samp_rate, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(0, 1)

        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0.enable_tags(False)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_NORM, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(True)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_0.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_0.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.display_grid_layout_0.addWidget(self._qtgui_time_sink_x_0_win, 0, 1, 5, 1)
        for r in range(0, 5):
            self.display_grid_layout_0.setRowStretch(r, 1)
        for c in range(1, 2):
            self.display_grid_layout_0.setColumnStretch(c, 1)
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
        self.qtgui_sink_x_0.set_update_time(1.0/0.666)
        self._qtgui_sink_x_0_win = sip.wrapinstance(self.qtgui_sink_x_0.qwidget(), Qt.QWidget)

        self.qtgui_sink_x_0.enable_rf_freq(True)

        self.display_grid_layout_0.addWidget(self._qtgui_sink_x_0_win, 0, 0, 5, 1)
        for r in range(0, 5):
            self.display_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 1):
            self.display_grid_layout_0.setColumnStretch(c, 1)
        self.freq_xlating_fir_filter_xxx_0_1 = filter.freq_xlating_fir_filter_ccc(1, firdes.complex_band_pass(1, samp_rate, -samp_rate/(2), samp_rate/(2), 10), 0, samp_rate)
        self.freq_xlating_fir_filter_xxx_0_1.set_block_alias("doppler_shift")
        self.epy_block_0 = epy_block_0.blk()
        self.blocks_vector_source_x_0_0 = blocks.vector_source_c(([0.0]*int(ud_spacing*samp_rate) + [np.exp(x/(riseFracc*ud_samples)) for x in range(-int(riseFracc*ud_samples), 0)]+[np.exp(-x/((1-riseFracc)*ud_samples)) for x in range(0, int((1-riseFracc)*ud_samples))]), True, 1, [])
        self.blocks_throttle2_0 = blocks.throttle( gr.sizeof_gr_complex*1, samp_rate, True, 0 if "auto" == "auto" else max( int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1) )
        self.blocks_null_sink_1 = blocks.null_sink(gr.sizeof_float*1)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(beacon_att)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_sig_source_x_0 = analog.sig_source_c(samp_rate, analog.GR_SIN_WAVE, beacon_freq, 1, 0, 0)
        self.analog_sig_source_x_0.set_block_alias("Beacon")
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, (noiseLevel/10), 0)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.epy_block_0, 'freq_out'), (self.freq_xlating_fir_filter_xxx_0_1, 'freq'))
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_0, 2))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_throttle2_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.freq_xlating_fir_filter_xxx_0_1, 0))
        self.connect((self.blocks_throttle2_0, 0), (self.qtgui_sink_x_0, 0))
        self.connect((self.blocks_throttle2_0, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.blocks_vector_source_x_0_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.epy_block_0, 0), (self.blocks_null_sink_1, 0))
        self.connect((self.freq_xlating_fir_filter_xxx_0_1, 0), (self.blocks_add_xx_0, 1))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "gr_meteor")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_ud_dur(self):
        return self.ud_dur

    def set_ud_dur(self, ud_dur):
        self.ud_dur = ud_dur
        self.set_ud_samples(self.samp_rate*self.ud_dur)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_samp_rate_label(self.samp_rate)
        self.set_ud_samples(self.samp_rate*self.ud_dur)
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)
        self.blocks_throttle2_0.set_sample_rate(self.samp_rate)
        self.blocks_vector_source_x_0_0.set_data(([0.0]*int(self.ud_spacing*self.samp_rate) + [np.exp(x/(self.riseFracc*self.ud_samples)) for x in range(-int(self.riseFracc*self.ud_samples), 0)]+[np.exp(-x/((1-self.riseFracc)*self.ud_samples)) for x in range(0, int((1-self.riseFracc)*self.ud_samples))]), [])
        self.freq_xlating_fir_filter_xxx_0_1.set_taps(firdes.complex_band_pass(1, self.samp_rate, -self.samp_rate/(2), self.samp_rate/(2), 10))
        self.qtgui_sink_x_0.set_frequency_range(self.beacon_freq, self.samp_rate)
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)

    def get_ud_spacing(self):
        return self.ud_spacing

    def set_ud_spacing(self, ud_spacing):
        self.ud_spacing = ud_spacing
        self.blocks_vector_source_x_0_0.set_data(([0.0]*int(self.ud_spacing*self.samp_rate) + [np.exp(x/(self.riseFracc*self.ud_samples)) for x in range(-int(self.riseFracc*self.ud_samples), 0)]+[np.exp(-x/((1-self.riseFracc)*self.ud_samples)) for x in range(0, int((1-self.riseFracc)*self.ud_samples))]), [])

    def get_ud_samples(self):
        return self.ud_samples

    def set_ud_samples(self, ud_samples):
        self.ud_samples = ud_samples
        self.blocks_vector_source_x_0_0.set_data(([0.0]*int(self.ud_spacing*self.samp_rate) + [np.exp(x/(self.riseFracc*self.ud_samples)) for x in range(-int(self.riseFracc*self.ud_samples), 0)]+[np.exp(-x/((1-self.riseFracc)*self.ud_samples)) for x in range(0, int((1-self.riseFracc)*self.ud_samples))]), [])

    def get_samp_rate_label(self):
        return self.samp_rate_label

    def set_samp_rate_label(self, samp_rate_label):
        self.samp_rate_label = samp_rate_label
        Qt.QMetaObject.invokeMethod(self._samp_rate_label_label, "setText", Qt.Q_ARG("QString", str(self._samp_rate_label_formatter(self.samp_rate_label))))

    def get_riseFracc(self):
        return self.riseFracc

    def set_riseFracc(self, riseFracc):
        self.riseFracc = riseFracc
        self.blocks_vector_source_x_0_0.set_data(([0.0]*int(self.ud_spacing*self.samp_rate) + [np.exp(x/(self.riseFracc*self.ud_samples)) for x in range(-int(self.riseFracc*self.ud_samples), 0)]+[np.exp(-x/((1-self.riseFracc)*self.ud_samples)) for x in range(0, int((1-self.riseFracc)*self.ud_samples))]), [])

    def get_noiseLevel(self):
        return self.noiseLevel

    def set_noiseLevel(self, noiseLevel):
        self.noiseLevel = noiseLevel
        self.analog_noise_source_x_0.set_amplitude((self.noiseLevel/10))

    def get_beacon_freq(self):
        return self.beacon_freq

    def set_beacon_freq(self, beacon_freq):
        self.beacon_freq = beacon_freq
        Qt.QMetaObject.invokeMethod(self._beacon_freq_line_edit, "setText", Qt.Q_ARG("QString", eng_notation.num_to_str(self.beacon_freq)))
        self.analog_sig_source_x_0.set_frequency(self.beacon_freq)
        self.qtgui_sink_x_0.set_frequency_range(self.beacon_freq, self.samp_rate)

    def get_beacon_att(self):
        return self.beacon_att

    def set_beacon_att(self, beacon_att):
        self.beacon_att = beacon_att
        self.blocks_multiply_const_vxx_0.set_k(self.beacon_att)




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
