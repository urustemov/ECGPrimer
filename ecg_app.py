import os
import sys
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QPushButton, QLabel, QTextEdit, QVBoxLayout, QHBoxLayout,
    QInputDialog, QCheckBox, QComboBox, QGroupBox, QFormLayout
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from scipy.interpolate import UnivariateSpline

# PyWavelets is an optional dependency. The application keeps working as a
# pure smoothing-spline tool if it is not installed.
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    pywt = None
    PYWT_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Wavelet denoising helper
# --------------------------------------------------------------------------- #
def wavelet_denoise(signal, wavelet="sym5", level=4, mode="soft"):
    """
    Wavelet denoising via discrete wavelet transform (DWT) and
    Donoho-Johnstone universal thresholding.

    Parameters
    ----------
    signal : 1-D numpy array, uniformly sampled.
    wavelet : str
        Mother wavelet name (e.g. 'sym5', 'db4', 'coif3').
    level : int
        Number of DWT decomposition levels.
    mode : str
        'soft' or 'hard' thresholding.

    Returns
    -------
    denoised : 1-D numpy array of the same length as `signal`.
    info : dict with the threshold value, sigma estimate and used level.
    """
    if not PYWT_AVAILABLE:
        raise RuntimeError("PyWavelets is not installed.")

    n = len(signal)
    # Cap the level so wavedec is well-defined for short signals.
    max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    used_level = min(level, max_level)

    coeffs = pywt.wavedec(signal, wavelet, level=used_level)

    # Robust noise estimate: MAD of the finest-scale detail coefficients
    # divided by 0.6745 (Donoho & Johnstone, 1994).
    detail_finest = coeffs[-1]
    sigma = np.median(np.abs(detail_finest)) / 0.6745 if len(detail_finest) > 0 else 0.0

    # Universal threshold.
    threshold = sigma * np.sqrt(2.0 * np.log(n)) if n > 1 else 0.0

    # Threshold every detail level; keep approximation untouched.
    new_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]
    ]

    denoised = pywt.waverec(new_coeffs, wavelet)
    # waverec can return a signal of length n or n+1 depending on parity.
    denoised = denoised[:n]

    return denoised, {
        "sigma": float(sigma),
        "threshold": float(threshold),
        "level_used": int(used_level),
        "wavelet": wavelet,
        "mode": mode,
    }



# --------------------------------------------------------------------------- #
# Matplotlib canvas
# --------------------------------------------------------------------------- #
class ECGCanvas(FigureCanvas):
    def __init__(self, parent_window=None):
        self.fig = Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.parent_window = parent_window
        self.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        if self.parent_window is None:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.parent_window.handle_plot_click(event.xdata, event.ydata)


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class ECGDigitizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "ECG Digitizer + Wavelet Denoising: "
            "f(t), f'(t), f''(t), f'''(t)"
        )
        self.resize(1500, 900)

        # image
        self.img = None
        self.img_path = None

        # interaction mode
        self.mode = None  # None / cal_x / cal_y / trace
        self.cal_x_pts = []
        self.cal_y_pts = []
        self.curve_pts = []

        # calibration values
        self.t0 = None
        self.t1 = None
        self.u0 = None
        self.u1 = None

        # transformed data
        self.t_sec = None
        self.u_mV = None

        # computed arrays
        self.t_grid = None
        self.x_norm = None
        self.f_raw = None         # spline only, without wavelet denoising
        self.f = None             # final signal (raw or denoised)
        self.f1 = None
        self.f2 = None
        self.f3 = None

        # last wavelet info (for log / reporting in the thesis)
        self.last_wavelet_info = None

        # export figures
        self.fig_f = None
        self.fig_f1 = None
        self.fig_f2 = None
        self.fig_f3 = None
        self.fig_fn = None
        self.fig_f1n = None
        self.fig_f2n = None
        self.fig_f3n = None

        # comparison data
        self.compare_df = None

        self.setup_ui()

        self.write_log(
            "Шаги:\n"
            "1) Load ECG Image\n"
            "2) Calibrate X -> кликни 2 точки по оси/сетке времени и введи их значения\n"
            "3) Calibrate Y -> кликни 2 точки по оси/сетке амплитуды и введи их значения\n"
            "4) Trace Curve Points -> поставь точки по кривой\n"
            "5) (Optional) Enable Wavelet Denoising в правой панели\n"
            "6) Compute -> считает f, f', f'', f'''\n"
            "7) Export CSV + PNG\n"
            "8) Для сравнения двух снимков: обработай первый, экспортируй CSV; "
            "потом обработай второй и нажми Compare with Exported CSV\n"
        )
        if not PYWT_AVAILABLE:
            self.write_log(
                "ВНИМАНИЕ: PyWavelets не установлен. "
                "Wavelet-денойзинг отключён. Установка: pip install PyWavelets"
            )

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        right = QVBoxLayout()

        # --- main action buttons ---
        self.btn_load = QPushButton("Load ECG Image")
        self.btn_load.clicked.connect(self.load_image)
        left.addWidget(self.btn_load)

        self.btn_cal_x = QPushButton("Calibrate X")
        self.btn_cal_x.clicked.connect(self.start_cal_x)
        left.addWidget(self.btn_cal_x)

        self.btn_cal_y = QPushButton("Calibrate Y")
        self.btn_cal_y.clicked.connect(self.start_cal_y)
        left.addWidget(self.btn_cal_y)

        self.btn_trace = QPushButton("Trace Curve Points")
        self.btn_trace.clicked.connect(self.start_trace)
        left.addWidget(self.btn_trace)

        self.btn_compute = QPushButton("Compute f(t) ... f'''(t)")
        self.btn_compute.clicked.connect(self.compute_all)
        left.addWidget(self.btn_compute)

        self.btn_export = QPushButton("Export CSV + PNG")
        self.btn_export.clicked.connect(self.export_outputs)
        left.addWidget(self.btn_export)

        self.btn_compare = QPushButton("Compare with Exported CSV")
        self.btn_compare.clicked.connect(self.compare_with_csv)
        left.addWidget(self.btn_compare)

        self.btn_show_abs = QPushButton("Show Absolute Plot")
        self.btn_show_abs.clicked.connect(self.plot_absolute_main)
        left.addWidget(self.btn_show_abs)

        self.btn_show_norm = QPushButton("Show Normalized Plot")
        self.btn_show_norm.clicked.connect(self.plot_normalized_main)
        left.addWidget(self.btn_show_norm)

        # New: side-by-side comparison of spline vs spline+wavelet for the
        # currently traced ECG. Useful for the thesis figure.
        self.btn_show_compare = QPushButton("Show Spline vs Wavelet (current)")
        self.btn_show_compare.clicked.connect(self.plot_spline_vs_wavelet)
        left.addWidget(self.btn_show_compare)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_all)
        left.addWidget(self.btn_reset)

        # --- Wavelet group box ---
        wavelet_group = QGroupBox("Wavelet Denoising")
        wavelet_form = QFormLayout()

        self.chk_wavelet = QCheckBox("Enable wavelet denoising")
        self.chk_wavelet.setChecked(PYWT_AVAILABLE)
        self.chk_wavelet.setEnabled(PYWT_AVAILABLE)
        wavelet_form.addRow(self.chk_wavelet)

        self.combo_wavelet = QComboBox()
        self.combo_wavelet.addItems([
            "sym5", "sym4", "sym6", "sym8",
            "db4", "db6", "db8",
            "coif3", "coif5",
            "bior3.5", "bior4.4",
        ])
        self.combo_wavelet.setCurrentText("sym5")
        self.combo_wavelet.setEnabled(PYWT_AVAILABLE)
        wavelet_form.addRow(QLabel("Mother wavelet:"), self.combo_wavelet)

        self.combo_level = QComboBox()
        self.combo_level.addItems([str(i) for i in range(1, 9)])
        self.combo_level.setCurrentText("4")
        self.combo_level.setEnabled(PYWT_AVAILABLE)
        wavelet_form.addRow(QLabel("Decomposition level:"), self.combo_level)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["soft", "hard"])
        self.combo_mode.setCurrentText("soft")
        self.combo_mode.setEnabled(PYWT_AVAILABLE)
        wavelet_form.addRow(QLabel("Thresholding mode:"), self.combo_mode)

        wavelet_group.setLayout(wavelet_form)
        left.addWidget(wavelet_group)

        # --- log ---
        left.addWidget(QLabel("Log"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        left.addWidget(self.log_box, 1)

        # --- right: canvas ---
        self.canvas = ECGCanvas(parent_window=self)
        right.addWidget(self.canvas)

        root.addLayout(left, 0)
        root.addLayout(right, 1)

    def write_log(self, text: str):
        self.log_box.append(text)

    # ------------------------------------------------------------------ #
    # Mouse / interaction
    # ------------------------------------------------------------------ #
    def handle_plot_click(self, x, y):
        if self.mode == "cal_x":
            self.cal_x_pts.append((x, y))
            self.write_log(f"Calibrate X: point {len(self.cal_x_pts)} = ({x:.2f}, {y:.2f})")
            self.redraw_main_canvas()
            if len(self.cal_x_pts) == 2:
                v1, ok1 = QInputDialog.getDouble(
                    self, "X calibration", "Time value of point 1:", decimals=6
                )
                if not ok1:
                    self.cal_x_pts = []
                    self.mode = None
                    return
                v2, ok2 = QInputDialog.getDouble(
                    self, "X calibration", "Time value of point 2:", decimals=6
                )
                if not ok2:
                    self.cal_x_pts = []
                    self.mode = None
                    return
                self.t0, self.t1 = v1, v2
                self.mode = None
                self.write_log(f"X calibrated: pixel -> time using values {self.t0} and {self.t1}")
                self.redraw_main_canvas()

        elif self.mode == "cal_y":
            self.cal_y_pts.append((x, y))
            self.write_log(f"Calibrate Y: point {len(self.cal_y_pts)} = ({x:.2f}, {y:.2f})")
            self.redraw_main_canvas()
            if len(self.cal_y_pts) == 2:
                v1, ok1 = QInputDialog.getDouble(
                    self, "Y calibration", "Voltage value of point 1:", decimals=6
                )
                if not ok1:
                    self.cal_y_pts = []
                    self.mode = None
                    return
                v2, ok2 = QInputDialog.getDouble(
                    self, "Y calibration", "Voltage value of point 2:", decimals=6
                )
                if not ok2:
                    self.cal_y_pts = []
                    self.mode = None
                    return
                self.u0, self.u1 = v1, v2
                self.mode = None
                self.write_log(f"Y calibrated: pixel -> voltage using values {self.u0} and {self.u1}")
                self.redraw_main_canvas()

        elif self.mode == "trace":
            self.curve_pts.append((x, y))
            self.write_log(f"Trace point {len(self.curve_pts)} = ({x:.2f}, {y:.2f})")
            self.redraw_main_canvas()

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open ECG Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not file_path:
            return
        try:
            img = mpimg.imread(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load image:\n{e}")
            return
        self.img = img
        self.img_path = file_path
        self.write_log(f"Loaded image: {file_path}")
        self.redraw_main_canvas()

    def redraw_main_canvas(self):
        # Make sure we are on a single-axes layout. If the user previously
        # viewed a multi-subplot grid (normalized/absolute plots, comparison),
        # self.canvas.ax may point to one of the subplots and the figure
        # may have multiple axes. Clear the whole figure and re-create a
        # single axes for image display + clicking.
        if len(self.canvas.fig.axes) != 1:
            self.canvas.fig.clear()
            self.canvas.ax = self.canvas.fig.add_subplot(111)

        ax = self.canvas.ax
        ax.clear()
        if self.img is not None:
            ax.imshow(self.img)
            ax.set_title("ECG image")
        else:
            ax.set_title("No image loaded")

        if self.cal_x_pts:
            xs = [p[0] for p in self.cal_x_pts]
            ys = [p[1] for p in self.cal_x_pts]
            ax.plot(xs, ys, "ro", label="X calibration")
        if self.cal_y_pts:
            xs = [p[0] for p in self.cal_y_pts]
            ys = [p[1] for p in self.cal_y_pts]
            ax.plot(xs, ys, "go", label="Y calibration")
        if self.curve_pts:
            xs = [p[0] for p in self.curve_pts]
            ys = [p[1] for p in self.curve_pts]
            ax.plot(xs, ys, "bo-", markersize=3, linewidth=1, label="Traced curve")

        if self.cal_x_pts or self.cal_y_pts or self.curve_pts:
            ax.legend(loc="best")
        self.canvas.draw()

    def start_cal_x(self):
        if self.img is None:
            QMessageBox.warning(self, "Warning", "Load an image first.")
            return
        self.mode = "cal_x"
        self.cal_x_pts = []
        self.write_log("Mode: Calibrate X. Click 2 points on the image.")
        self.redraw_main_canvas()

    def start_cal_y(self):
        if self.img is None:
            QMessageBox.warning(self, "Warning", "Load an image first.")
            return
        self.mode = "cal_y"
        self.cal_y_pts = []
        self.write_log("Mode: Calibrate Y. Click 2 points on the image.")
        self.redraw_main_canvas()

    def start_trace(self):
        if self.img is None:
            QMessageBox.warning(self, "Warning", "Load an image first.")
            return
        self.mode = "trace"
        self.curve_pts = []
        self.write_log("Mode: Trace Curve Points. Click along ECG curve.")
        self.redraw_main_canvas()

    # ------------------------------------------------------------------ #
    # Pixel <-> physical coordinates
    # ------------------------------------------------------------------ #
    def pixel_to_time(self, x):
        if len(self.cal_x_pts) != 2 or self.t0 is None or self.t1 is None:
            raise ValueError("X calibration is incomplete.")
        x0 = self.cal_x_pts[0][0]
        x1 = self.cal_x_pts[1][0]
        if abs(x1 - x0) < 1e-12:
            raise ValueError("Invalid X calibration points.")
        return self.t0 + (x - x0) * (self.t1 - self.t0) / (x1 - x0)

    def pixel_to_voltage(self, y):
        if len(self.cal_y_pts) != 2 or self.u0 is None or self.u1 is None:
            raise ValueError("Y calibration is incomplete.")
        y0 = self.cal_y_pts[0][1]
        y1 = self.cal_y_pts[1][1]
        if abs(y1 - y0) < 1e-12:
            raise ValueError("Invalid Y calibration points.")
        return self.u0 + (y - y0) * (self.u1 - self.u0) / (y1 - y0)

    # ------------------------------------------------------------------ #
    # Core computation
    # ------------------------------------------------------------------ #
    def compute_all(self):
        try:
            if len(self.curve_pts) < 6:
                QMessageBox.warning(self, "Warning", "Trace at least 6 points first.")
                return

            # Sanity check on X calibration: the two X-calibration points
            # must correspond to different time values, otherwise every
            # traced point collapses to nearly the same t.
            if self.t0 is not None and self.t1 is not None:
                if abs(self.t1 - self.t0) < 1e-9:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        "X calibration values are equal (t0 == t1).\n"
                        "Re-run Calibrate X and enter two DIFFERENT time values "
                        "(e.g. 0 and 1 seconds)."
                    )
                    return
            if self.u0 is not None and self.u1 is not None:
                if abs(self.u1 - self.u0) < 1e-9:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        "Y calibration values are equal (u0 == u1).\n"
                        "Re-run Calibrate Y and enter two DIFFERENT voltage values."
                    )
                    return

            pts = np.array(self.curve_pts, dtype=float)
            t = np.array([self.pixel_to_time(x) for x, _ in pts], dtype=float)
            u = np.array([self.pixel_to_voltage(y) for _, y in pts], dtype=float)

            n_input = len(t)

            # sort by time
            order = np.argsort(t)
            t = t[order]
            u = u[order]

            # remove duplicate t
            t_unique = [t[0]]
            u_unique = [u[0]]
            for i in range(1, len(t)):
                if abs(t[i] - t_unique[-1]) > 1e-12:
                    t_unique.append(t[i])
                    u_unique.append(u[i])
            t = np.array(t_unique, dtype=float)
            u = np.array(u_unique, dtype=float)

            n_unique = len(t)
            n_collapsed = n_input - n_unique

            if n_unique < 6:
                # Build a diagnostic message so the user knows WHY this failed.
                msg_lines = [
                    f"Need at least 6 unique time points, got {n_unique}.",
                    "",
                    f"Traced points clicked:    {len(self.curve_pts)}",
                    f"Unique after calibration: {n_unique}",
                    f"Collapsed to duplicates:  {n_collapsed}",
                ]
                if n_collapsed > 0:
                    msg_lines.append("")
                    msg_lines.append(
                        "Many traced points map to the same time value. "
                        "Most likely your X calibration values are wrong or "
                        "your two X-calibration clicks are too close in pixels."
                    )
                if len(self.curve_pts) < 6:
                    msg_lines.append("")
                    msg_lines.append(
                        "You also clicked fewer than 6 trace points — add more."
                    )
                QMessageBox.warning(self, "Warning", "\n".join(msg_lines))
                self.write_log("\n".join(msg_lines))
                return

            self.t_sec = t
            self.u_mV = u

            t_grid = np.linspace(t.min(), t.max(), 1000)

            # Stage 1: smoothing spline on the discrete traced points.
            # This step turns ~20-80 manually clicked points into a uniformly
            # sampled signal on t_grid (1000 points).
            s_factor = 0.01
            s = s_factor * len(t)
            spl = UnivariateSpline(t, u, s=s, k=5)
            self.t_grid = t_grid
            self.f_raw = spl(t_grid)

            # Stage 2: optional wavelet denoising on the uniformly sampled
            # signal. The wavelet transform is mathematically well-defined
            # here because t_grid is uniform.
            wavelet_used = (
                self.chk_wavelet.isChecked()
                and PYWT_AVAILABLE
            )
            if wavelet_used:
                wavelet_name = self.combo_wavelet.currentText()
                level = int(self.combo_level.currentText())
                mode = self.combo_mode.currentText()
                self.f, info = wavelet_denoise(
                    self.f_raw, wavelet=wavelet_name, level=level, mode=mode
                )
                self.last_wavelet_info = info
                self.write_log(
                    "Wavelet denoising applied: "
                    f"wavelet={info['wavelet']}, level={info['level_used']}, "
                    f"mode={info['mode']}, sigma={info['sigma']:.6g}, "
                    f"threshold={info['threshold']:.6g}"
                )
            else:
                self.f = self.f_raw.copy()
                self.last_wavelet_info = None
                if self.chk_wavelet.isChecked() and not PYWT_AVAILABLE:
                    self.write_log("PyWavelets is unavailable; using raw spline only.")
                else:
                    self.write_log("Wavelet denoising disabled; using raw spline only.")

            # Stage 3: derivatives.
            #
            # Direct analytic differentiation of an interpolating quintic
            # spline. This gives smooth, physically reasonable shapes for
            # f', f'', f'''. Derivatives of order 4 and higher are not
            # computed here: on the typical sampling step h ~ 8e-5 s for
            # manually traced ECG fragments, the noise amplification
            # factor h^{-k} exceeds 10^16 by k=4 and exceeds 10^20 by
            # k=5, dominating any genuine signal. Regularized estimation
            # of f^(4), f^(5) is left as a direction for future work
            # (see Chapter 7, Limitations and Future Work).
            spl_for_deriv = UnivariateSpline(t_grid, self.f, s=0, k=5)
            self.f1 = spl_for_deriv.derivative(1)(t_grid)
            self.f2 = spl_for_deriv.derivative(2)(t_grid)
            self.f3 = spl_for_deriv.derivative(3)(t_grid)

            # Normalized x coordinate from 0 to 1.
            t_start = t_grid[0]
            t_end = t_grid[-1]
            if abs(t_end - t_start) < 1e-12:
                QMessageBox.warning(self, "Warning", "Invalid time range after calibration.")
                return
            self.x_norm = (t_grid - t_start) / (t_end - t_start)

            self.build_all_figures()
            self.plot_normalized_main()
            self.write_log(
                "Computed successfully:\n"
                "f(t), f'(t), f''(t), f'''(t)\n"
                "Normalized coordinate x_norm in [0, 1] created for comparison."
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Computation failed:\n{e}")

    # ------------------------------------------------------------------ #
    # Figure building
    # ------------------------------------------------------------------ #
    def build_single_figure(self, x, y, title, xlabel, ylabel):
        fig = Figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        return fig

    def build_all_figures(self):
        # absolute
        self.fig_f = self.build_single_figure(
            self.t_grid, self.f, "f(t)", "t (s)", "U"
        )
        self.fig_f1 = self.build_single_figure(
            self.t_grid, self.f1, "f'(t)", "t (s)", "dU/dt"
        )
        self.fig_f2 = self.build_single_figure(
            self.t_grid, self.f2, "f''(t)", "t (s)", "d²U/dt²"
        )
        self.fig_f3 = self.build_single_figure(
            self.t_grid, self.f3, "f'''(t)", "t (s)", "d³U/dt³"
        )
        # normalized
        self.fig_fn = self.build_single_figure(
            self.x_norm, self.f, "f(x_norm)", "x_norm", "U"
        )
        self.fig_f1n = self.build_single_figure(
            self.x_norm, self.f1, "f'(x_norm)", "x_norm", "dU/dt"
        )
        self.fig_f2n = self.build_single_figure(
            self.x_norm, self.f2, "f''(x_norm)", "x_norm", "d²U/dt²"
        )
        self.fig_f3n = self.build_single_figure(
            self.x_norm, self.f3, "f'''(x_norm)", "x_norm", "d³U/dt³"
        )

    # ------------------------------------------------------------------ #
    # Plot helpers
    # ------------------------------------------------------------------ #
    def _plot_grid(self, x, x_label, title_suffix):
        """
        Helper: draw f, f', f'', f''' on a 2x2 grid of subplots on the
        main canvas. Each subplot has its own y-scale.
        """
        if x is None or self.f is None:
            QMessageBox.warning(self, "Warning", "Compute data first.")
            return

        fig = self.canvas.fig
        fig.clear()
        axes = fig.subplots(2, 2, sharex=True)
        flat = axes.ravel()

        series = [
            (self.f,  "f",     "U"),
            (self.f1, "f'",    "dU/dt"),
            (self.f2, "f''",   "d²U/dt²"),
            (self.f3, "f'''",  "d³U/dt³"),
        ]
        for ax, (y, name, ylabel) in zip(flat, series):
            ax.plot(x, y, linewidth=1.2)
            ax.set_title(name, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.4)
            ax.tick_params(labelsize=8)

        # x-axis labels only on the bottom row.
        for ax in axes[-1, :]:
            ax.set_xlabel(x_label, fontsize=9)

        fig.suptitle(title_suffix, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        # IMPORTANT: keep self.canvas.ax pointing at a valid axes for
        # later interactive use (load image, etc.).
        self.canvas.ax = flat[0]
        self.canvas.draw()

    def plot_absolute_main(self):
        self._plot_grid(self.t_grid, "t (s)", "Absolute coordinates")

    def plot_normalized_main(self):
        self._plot_grid(self.x_norm, "x_norm (0 = start, 1 = end)",
                        "Normalized coordinates for comparison")

    def plot_spline_vs_wavelet(self):
        """
        Compare the raw smoothing-spline output against the
        wavelet-denoised version on the same axes.
        Useful for the Results chapter of the thesis.
        """
        if self.f_raw is None or self.f is None or self.t_grid is None:
            QMessageBox.warning(self, "Warning", "Compute data first.")
            return

        # Restore a single-axes layout for this view.
        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        self.canvas.ax = ax

        ax.plot(self.t_grid, self.f_raw, label="Spline only (f_raw)", alpha=0.7)
        ax.plot(self.t_grid, self.f, label="Spline + Wavelet (f)", linewidth=1.4)
        title = "Spline vs Spline+Wavelet"
        if self.last_wavelet_info is not None:
            info = self.last_wavelet_info
            title += (
                f"  [{info['wavelet']}, level {info['level_used']}, "
                f"{info['mode']}, threshold={info['threshold']:.4g}]"
            )
        else:
            title += "  [wavelet OFF — кривые совпадают]"
        ax.set_title(title)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("U")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        self.canvas.draw()

        # Quick numeric summary
        diff = self.f - self.f_raw
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        max_abs = float(np.max(np.abs(diff)))
        self.write_log(
            "Difference between raw spline and wavelet-denoised signal:\n"
            f"  RMSE    = {rmse:.6g}\n"
            f"  Max|Δ|  = {max_abs:.6g}"
        )

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def export_outputs(self):
        if self.t_grid is None or self.f is None:
            QMessageBox.warning(self, "Warning", "Compute data first.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not out_dir:
            return

        try:
            base_name = "ecg_export"
            if self.img_path:
                stem = os.path.splitext(os.path.basename(self.img_path))[0]
                base_name = f"{stem}_export"

            # Attach a wavelet tag to the filename so the two regimes don't
            # overwrite each other during A/B experiments.
            if self.last_wavelet_info is not None:
                info = self.last_wavelet_info
                base_name += f"_wt-{info['wavelet']}-L{info['level_used']}-{info['mode']}"
            else:
                base_name += "_no-wt"

            csv_path = os.path.join(out_dir, f"{base_name}.csv")

            df_dict = {
                "t_sec": self.t_grid,
                "x_norm": self.x_norm,
                "f": self.f,
                "f1": self.f1,
                "f2": self.f2,
                "f3": self.f3,
            }
            # Also include the pre-wavelet signal for completeness.
            if self.f_raw is not None:
                df_dict["f_raw_spline_only"] = self.f_raw
            df = pd.DataFrame(df_dict)
            df.to_csv(csv_path, index=False)

            self.fig_f.savefig(os.path.join(out_dir, f"{base_name}_f.png"), dpi=200, bbox_inches="tight")
            self.fig_f1.savefig(os.path.join(out_dir, f"{base_name}_f1.png"), dpi=200, bbox_inches="tight")
            self.fig_f2.savefig(os.path.join(out_dir, f"{base_name}_f2.png"), dpi=200, bbox_inches="tight")
            self.fig_f3.savefig(os.path.join(out_dir, f"{base_name}_f3.png"), dpi=200, bbox_inches="tight")

            self.fig_fn.savefig(os.path.join(out_dir, f"{base_name}_norm_f.png"), dpi=200, bbox_inches="tight")
            self.fig_f1n.savefig(os.path.join(out_dir, f"{base_name}_norm_f1.png"), dpi=200, bbox_inches="tight")
            self.fig_f2n.savefig(os.path.join(out_dir, f"{base_name}_norm_f2.png"), dpi=200, bbox_inches="tight")
            self.fig_f3n.savefig(os.path.join(out_dir, f"{base_name}_norm_f3.png"), dpi=200, bbox_inches="tight")

            # Also save a small textual report — handy for the thesis.
            report_path = os.path.join(out_dir, f"{base_name}_report.txt")
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write("ECG Digitization Report\n")
                fh.write("=======================\n\n")
                fh.write(f"Source image: {self.img_path}\n")
                fh.write(f"Number of traced points: {len(self.curve_pts)}\n")
                fh.write(f"Time range: [{self.t_grid[0]:.4f}, {self.t_grid[-1]:.4f}] s\n")
                fh.write(f"Grid size: {len(self.t_grid)} samples\n\n")
                fh.write("Smoothing spline parameters:\n")
                fh.write("  k = 5 (quintic)\n")
                fh.write("  s = 0.01 * N (initial), s = 0 (post-wavelet)\n\n")
                if self.last_wavelet_info is not None:
                    info = self.last_wavelet_info
                    fh.write("Wavelet denoising parameters:\n")
                    fh.write(f"  wavelet     = {info['wavelet']}\n")
                    fh.write(f"  level_used  = {info['level_used']}\n")
                    fh.write(f"  mode        = {info['mode']}\n")
                    fh.write(f"  sigma (MAD) = {info['sigma']:.6g}\n")
                    fh.write(f"  threshold   = {info['threshold']:.6g}\n")
                else:
                    fh.write("Wavelet denoising: DISABLED\n")
                if self.f_raw is not None and self.f is not None:
                    diff = self.f - self.f_raw
                    fh.write(f"\nRMSE(f - f_raw)   = {float(np.sqrt(np.mean(diff ** 2))):.6g}\n")
                    fh.write(f"Max|f - f_raw|    = {float(np.max(np.abs(diff))):.6g}\n")

            QMessageBox.information(
                self,
                "Export done",
                f"Saved:\n{csv_path}\n{report_path}\n\n"
                "Absolute and normalized PNG graphs were also saved."
            )
            self.write_log(f"Exported CSV, PNG and report to: {out_dir}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{e}")

    # ------------------------------------------------------------------ #
    # Compare with previously exported CSV
    # ------------------------------------------------------------------ #
    def compare_with_csv(self):
        if self.x_norm is None or self.f is None:
            QMessageBox.warning(self, "Warning", "Compute current image first.")
            return

        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open exported CSV for comparison",
            "",
            "CSV files (*.csv)"
        )
        if not csv_path:
            return

        try:
            df2 = pd.read_csv(csv_path)

            required_cols = {"x_norm", "f", "f1", "f2", "f3"}
            if not required_cols.issubset(set(df2.columns)):
                raise ValueError(
                    "CSV does not contain required columns: "
                    "x_norm, f, f1, f2, f3"
                )

            self.compare_df = df2.copy()
            common_x = np.linspace(0.0, 1.0, 400)

            cur_f = np.interp(common_x, self.x_norm, self.f)
            cur_f1 = np.interp(common_x, self.x_norm, self.f1)
            cur_f2 = np.interp(common_x, self.x_norm, self.f2)
            cur_f3 = np.interp(common_x, self.x_norm, self.f3)

            ref_x = df2["x_norm"].to_numpy(dtype=float)
            ref_f = np.interp(common_x, ref_x, df2["f"].to_numpy(dtype=float))
            ref_f1 = np.interp(common_x, ref_x, df2["f1"].to_numpy(dtype=float))
            ref_f2 = np.interp(common_x, ref_x, df2["f2"].to_numpy(dtype=float))
            ref_f3 = np.interp(common_x, ref_x, df2["f3"].to_numpy(dtype=float))

            self.canvas.fig.clear()
            axes = self.canvas.fig.subplots(2, 2, sharex=True)
            flat = axes.ravel()

            cur_series = [cur_f, cur_f1, cur_f2, cur_f3]
            ref_series = [ref_f, ref_f1, ref_f2, ref_f3]
            names = ["f", "f'", "f''", "f'''"]

            for ax, cur_y, ref_y, name in zip(flat, cur_series, ref_series, names):
                ax.plot(common_x, cur_y, label="Current", linewidth=1.2)
                ax.plot(common_x, ref_y, label="Reference", linewidth=1.0, alpha=0.8)
                ax.set_title(name, fontsize=10)
                ax.grid(True, alpha=0.4)
                ax.tick_params(labelsize=8)

            flat[0].legend(loc="best", fontsize=8)
            for ax in axes[-1, :]:
                ax.set_xlabel("x_norm (0 = start, 1 = end)", fontsize=9)

            self.canvas.fig.suptitle(
                "Comparison on common normalized grid", fontsize=11
            )
            self.canvas.fig.tight_layout(rect=(0, 0, 1, 0.96))
            self.canvas.ax = flat[0]
            self.canvas.draw()

            self.write_log(
                f"Comparison loaded from: {csv_path}\n"
                "Both signals were resampled to the same normalized grid [0, 1]."
            )

            diff_f = np.max(np.abs(cur_f - ref_f))
            diff_f1 = np.max(np.abs(cur_f1 - ref_f1))
            diff_f2 = np.max(np.abs(cur_f2 - ref_f2))
            diff_f3 = np.max(np.abs(cur_f3 - ref_f3))

            rmse_f = float(np.sqrt(np.mean((cur_f - ref_f) ** 2)))
            corr_f = float(np.corrcoef(cur_f, ref_f)[0, 1])

            self.write_log(
                "Max absolute differences on common normalized grid:\n"
                f"f    : {diff_f:.6f}\n"
                f"f'   : {diff_f1:.6f}\n"
                f"f''  : {diff_f2:.6f}\n"
                f"f''' : {diff_f3:.6f}\n"
                f"RMSE(f) = {rmse_f:.6f}\n"
                f"Pearson r(f) = {corr_f:.6f}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Comparison failed:\n{e}")

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset_all(self):
        self.img = None
        self.img_path = None
        self.mode = None
        self.cal_x_pts = []
        self.cal_y_pts = []
        self.curve_pts = []
        self.t0 = None
        self.t1 = None
        self.u0 = None
        self.u1 = None
        self.t_sec = None
        self.u_mV = None
        self.t_grid = None
        self.x_norm = None
        self.f_raw = None
        self.f = None
        self.f1 = None
        self.f2 = None
        self.f3 = None
        self.last_wavelet_info = None
        self.fig_f = None
        self.fig_f1 = None
        self.fig_f2 = None
        self.fig_f3 = None
        self.fig_fn = None
        self.fig_f1n = None
        self.fig_f2n = None
        self.fig_f3n = None
        self.compare_df = None
        self.log_box.clear()
        self.write_log("Reset done.")
        self.redraw_main_canvas()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ECGDigitizer()
    win.show()
    sys.exit(app.exec())