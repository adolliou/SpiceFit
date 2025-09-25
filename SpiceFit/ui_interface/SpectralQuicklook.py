import copy
from ..core import FitResults, SpiceRaster, FittingModel, SpiceRasterWindowL2
from ..util.plotting_fits import PlotFits
from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QComboBox,
    QMainWindow,
    QTextEdit,
    QVBoxLayout,
    QStackedLayout,
    QPushButton,
    QHBoxLayout,
)
from PyQt5.QtGui import QColor, QPalette, QTransform
from PyQt5.QtWidgets import QWidget
from functools import partial
import os
from pathlib import Path
from astropy.visualization import (
    ImageNormalize,
    AsymmetricPercentileInterval,
    SqrtStretch,
    LinearStretch,
    LogStretch,
)
from matplotlib.backends.backend_qt5agg import FigureCanvas
from ..util.constants import Constants
import matplotlib as mpl
from matplotlib.patches import Rectangle
import astropy.units as u


class GUISpectralQuicklook(QMainWindow):
    def __init__(self, fit_results: FitResults):
        super().__init__()

        self.fit_results = fit_results
        self.lines_list = list(self.fit_results.components_results.keys())
        self.lines_list.remove("main")
        self.lines_list.remove("flagged_pixels")

        self.set_up_layouts()
        self._set_UI_component()
        self._set_spectral_map(xpos=0, ypos=0, line_name=self.lines_list[0], param="radiance")
        self._set_spectral_plot(xpos=0, ypos=0)

        widget = QWidget()
        widget.setLayout(self.pagelayout)
        self.setCentralWidget(widget)

    def set_up_layouts(self):
        self.pagelayout = QVBoxLayout()
        self.button_layout = QHBoxLayout()
        self.text_layout = QHBoxLayout()
        self.images_layout = QHBoxLayout()

        self.pagelayout.addLayout(self.button_layout)
        self.pagelayout.addLayout(self.text_layout)
        self.pagelayout.addLayout(self.images_layout)

    def _set_UI_component(self):
        self.setWindowTitle("Spectral checker")

        self.xaxis_edit = QLineEdit("0")
        self.xaxis_edit.textChanged.connect(self.text_changed)
        self.xaxis_edit.returnPressed.connect(self.update_spectra)
        self.xaxis_edit.setMaxLength(10)
        self.xaxis_edit.setPlaceholderText("Pixel-X")
        self.button_layout.addWidget(self.xaxis_edit)

        self.yaxis_edit = QLineEdit("0")
        self.yaxis_edit.textChanged.connect(self.text_changed)
        self.yaxis_edit.returnPressed.connect(self.update_spectra)
        self.yaxis_edit.setMaxLength(10)
        self.yaxis_edit.setPlaceholderText("Pixel-Y:")
        self.button_layout.addWidget(self.yaxis_edit)

        self._update_param_list(line_name=self.lines_list[0])

        self.lines_list_widget = QComboBox()
        self.lines_list_widget.addItems(self.lines_list)
        self.lines_list_widget.currentTextChanged.connect(self.lines_list_widget_changed)
        self.button_layout.addWidget(self.lines_list_widget)

        btn = QPushButton("select")
        btn.pressed.connect(self.update_spectra)
        self.button_layout.addWidget(btn)

        # self.xlabel_label = QLabel("Pixel-X:")
        # self.text_layout.addWidget(self.xlabel_label)

        # self.ylabel_label = QLabel("Pixel-Y:")
        # self.text_layout.addWidget(self.ylabel_label)

    def _set_spectral_map(self, xpos, ypos, line_name,param="radiance"):

        cm = Constants.inch_to_cm
        self.figure_spectral_map = plt.figure(figsize=(17 * cm, 9 * cm))
        self.canvas_spectral_map = FigureCanvas(self.figure_spectral_map)
        self.ax_spectral_map = self.figure_spectral_map.subplots()

        self.fit_results.plot_fitted_map(fig=self.figure_spectral_map, ax=self.ax_spectral_map, 
                                         line=line_name, param=param, pixels= (xpos, ypos), doppler_mediansubtraction=True)
        self.images_layout.addWidget(self.canvas_spectral_map)

    def _set_spectral_plot(self, xpos, ypos):

        data_cube = self.fit_results.spectral_window.data
        uncertainty_cube = self.fit_results.spectral_window.uncertainty["Total"]
        xx, yy, ll, tt = self.fit_results.spectral_window.return_point_pixels()
        coords, lambda_cube, t = self.fit_results.spectral_window.wcs.pixel_to_world(
            xx, yy, ll, tt
        )

        data_l2 = np.squeeze(data_cube.to(Constants.conventional_spectral_units).value)
        uncertainty_l2 = np.squeeze(uncertainty_cube.to(
            Constants.conventional_spectral_units
        ).value)
        lambda_l2 = np.squeeze(lambda_cube.to(Constants.conventional_lambda_units).value)

        cm = Constants.inch_to_cm
        self.figure_spectral_plot = plt.figure(figsize=(17 * cm, 9 * cm))
        self.canvas_spectral_plot = FigureCanvas(self.figure_spectral_plot)
        self.ax_spectral_plot = self.figure_spectral_plot.subplots()
        self.ax_spectral_plot.errorbar(
            x=lambda_l2[:, ypos, xpos],
            y=data_l2[:, ypos, xpos],
            yerr=0.5 * uncertainty_l2[:, ypos, xpos],
            lw=0.9,
            marker="",
            linestyle="-",
            elinewidth=0.4,
            color="k",
            label="data",
        )
        yfit_total = self.fit_results.get_fitted_spectra(x=u.Quantity(lambda_l2[:, ypos, xpos],
                                                            Constants.conventional_lambda_units),
                                                position=(0, ypos, xpos),
                                                component="total")
        self.ax_spectral_plot.plot(
            lambda_l2[:, ypos, xpos],
            yfit_total,
            lw=0.9,
            marker="",
            linestyle="-",
            color="b",
            label="total",
        )

        self.ax_spectral_plot.set_xlabel(f"Wavelength [{Constants.conventional_lambda_units}]")
        self.ax_spectral_plot.set_ylabel(f"Spectra [{Constants.conventional_spectral_units}]")
        title = f"{self.lines_list[0]} : {str(xpos), str(ypos)}, "
        for param in self.fit_results.components_results["main"]["coeffs"].keys():
            if (param != "radiance") and (param != "velocity") and (param != "s"):
                title += f"{param} : {self.fit_results.components_results['main']['coeffs'][param]['results'].value[0,ypos, xpos]:.2f}, "
        if "background" in self.lines_list:
            title += f"; bcg: "
            for param in self.fit_results.components_results["background"]["coeffs"].keys():
                title += f"{param} : {self.fit_results.components_results['background']['coeffs'][param]['results'].value[0,ypos, xpos]:.2f}, "

        self.ax_spectral_plot.set_title(title)

        self.images_layout.addWidget(self.canvas_spectral_plot)

    def lines_list_widget_changed(self, s):
        self.param_list_widget.close()
        self._update_param_list(str(s))

    def text_changed(self, s):
        pass

    def text_edited(self, s):
        pass

    @property
    def ii(self):
        return self.iv.getImageItem()

    def _update_param_list(self, line_name):

        self.param_list_widget = QComboBox()
        list_params = self.fit_results.components_results[line_name]["coeffs"].keys()
        self.param_list_widget.addItems(list_params)
        self.button_layout.addWidget(self.param_list_widget)

    def update_spectra(self):
        sx = int(self.xaxis_edit.text())
        sy = int(self.yaxis_edit.text())
        print(f"changed pixels to ({sx}, {sy})")
        line_name = str(self.lines_list_widget.currentText())
        param = str(self.param_list_widget.currentText())

        self.canvas_spectral_map.close()
        self._set_spectral_map(sx, sy, line_name=line_name, param=param)

        self.canvas_spectral_plot.close()
        self._set_spectral_plot(sx, sy)

        self.param_list_widget.close()
        self._update_param_list(line_name)

        self.images_layout.update()


def launch_spectral_quicklook(res):
    app = QApplication(sys.argv)

    window = GUISpectralQuicklook(res)
    window.show()

    app.exec()


if __name__ == "__main__":
    pass
