import sys
import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox
)
from PySide6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scipy.interpolate import UnivariateSpline


PAPER_SPEED_MM_PER_S = 25.0   # 25 mm/s (not strictly needed in this calibration approach)
GAIN_MM_PER_MV = 5.0          # 5 mm/mV  (not strictly needed in this calibration approach)


class ECGApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Digitizer: f(t), f'(t), f''(t), f'''(t)")

        # Data holders
        self.img = None
        self.img_path = None

        self.cal_x = []  # [(px, t_value), (px, t_value)]
        self.cal_y = []  # [(py, mv_value), (py, mv_value)]
        self.curve_pts = []  # [(px, py), ...]

        self.mode = "idle"  # "cal_x", "cal_y", "curve"

        # UI
        central = QWidget()
        self.setCentralWidget(central)

        self.status = QLabel("1) Load image. Then click 'Calibrate X', 'Calibrate Y', 'Trace curve'.")
        self.btn_load = QPushButton("Load ECG image")
        self.btn_calx = QPushButton("Calibrate X (time)")
        self.btn_caly = QPushButton("Calibrate Y (mV)")
        self.btn_trace = QPushButton("Trace curve points")
        self.btn_compute = QPushButton("Compute f(t), f', f'', f'''")
        self.btn_export = QPushButton("Export CSV + PNG plots")
        self.btn_reset = QPushButton("Reset")

        self.btn_calx.setEnabled(False)
        self.btn_caly.setEnabled(False)
        self.btn_trace.setEnabled(False)
        self.btn_compute.setEnabled(False)
        self.btn_export.setEnabled(False)

        # Matplotlib Figure
        self.fig = Figure(figsize=(10, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        # IMPORTANT: make canvas receive keyboard focus so Enter/Return works
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()

        # Layouts
        left = QVBoxLayout()
        left.addWidget(self.btn_load)
        left.addWidget(self.btn_calx)
        left.addWidget(self.btn_caly)
        left.addWidget(self.btn_trace)
        left.addWidget(self.btn_compute)
        left.addWidget(self.btn_export)
        left.addWidget(self.btn_reset)
        left.addStretch()
        left.addWidget(self.status)

        root = QHBoxLayout()
        root.addLayout(left, 1)
        root.addWidget(self.canvas, 4)
        central.setLayout(root)

        # Events
        self.btn_load.clicked.connect(self.load_image)
        self.btn_calx.clicked.connect(self.set_mode_calx)
        self.btn_caly.clicked.connect(self.set_mode_caly)
        self.btn_trace.clicked.connect(self.set_mode_trace)
        self.btn_compute.clicked.connect(self.compute_all)
        self.btn_export.clicked.connect(self.export_outputs)
        self.btn_reset.clicked.connect(self.reset_all)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("key_press_event", self.on_key)

        # Computed results
        self.t_sec = None
        self.u_mV = None
        self.t_grid = None
        self.f = None
        self.f1 = None
        self.f2 = None
        self.f3 = None

        # Separate figures for outputs
        self.fig_f = None
        self.fig_f1 = None
        self.fig_f2 = None
        self.fig_f3 = None

        self.ax.clear()
        self.ax.set_title("Load image to start")
        self.ax.axis("off")
        self.canvas.draw()

    def msg(self, text: str):
        self.status.setText(text)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open ECG Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return

        import matplotlib.image as mpimg
        self.img = mpimg.imread(path)
        self.img_path = path

        self.ax.clear()
        self.ax.imshow(self.img)
        self.ax.set_title("ECG image (click to calibrate/trace)")
        self.ax.axis("off")
        self.canvas.draw()

        self.btn_calx.setEnabled(True)
        self.btn_caly.setEnabled(True)
        self.btn_trace.setEnabled(True)
        self.btn_compute.setEnabled(False)
        self.btn_export.setEnabled(False)

        self.msg("Image loaded. Step 2: Calibrate X (click 2 points on grid: 0 and 0.2 sec).")

        # ensure keyboard focus
        self.canvas.setFocus()

    def set_mode_calx(self):
        self.mode = "cal_x"
        self.cal_x = []
        self.btn_compute.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.msg("CAL X: click 2 points on TWO adjacent bold vertical grid lines. First = 0 sec, second = 0.2 sec.")
        self.canvas.setFocus()

    def set_mode_caly(self):
        self.mode = "cal_y"
        self.cal_y = []
        self.btn_compute.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.msg("CAL Y: click 2 points on TWO adjacent bold horizontal grid lines. First = 0 mV, second = +1 mV.")
        self.canvas.setFocus()

    def set_mode_trace(self):
        if len(self.cal_x) < 2 or len(self.cal_y) < 2:
            QMessageBox.warning(self, "Need calibration", "Calibrate X and Y first.")
            return
        self.mode = "curve"
        self.curve_pts = []
        self.btn_compute.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.msg("TRACE: click many points along ECG curve. Press ENTER/RETURN when done.")
        self.canvas.setFocus()

    def on_click(self, event):
        if self.img is None or event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)

        if self.mode == "cal_x":
            self.cal_x.append((x, None))
            self.ax.plot([x], [y], "ro")
            self.canvas.draw()
            self.canvas.setFocus()

            if len(self.cal_x) == 1:
                self.msg("CAL X: first point set (0 sec). Click second point (0.2 sec).")
            elif len(self.cal_x) == 2:
                # Assign values: 0 and 0.2
                self.cal_x[0] = (self.cal_x[0][0], 0.0)
                self.cal_x[1] = (self.cal_x[1][0], 0.2)
                self.mode = "idle"
                self.msg("CAL X done. Now Calibrate Y, then Trace curve.")

        elif self.mode == "cal_y":
            self.cal_y.append((y, None))
            self.ax.plot([x], [y], "go")
            self.canvas.draw()
            self.canvas.setFocus()

            if len(self.cal_y) == 1:
                self.msg("CAL Y: first point set (0 mV). Click second point (+1 mV).")
            elif len(self.cal_y) == 2:
                self.cal_y[0] = (self.cal_y[0][0], 0.0)
                self.cal_y[1] = (self.cal_y[1][0], 1.0)
                self.mode = "idle"
                self.msg("CAL Y done. Now Trace curve points.")

        elif self.mode == "curve":
            self.curve_pts.append((x, y))
            self.ax.plot([x], [y], "y.", markersize=4)
            self.canvas.draw()
            self.canvas.setFocus()

    def on_key(self, event):
        # Matplotlib often reports main Enter as "return" (not "enter")
        if self.mode == "curve" and (event.key in ("enter", "return")):
            if len(self.curve_pts) < 10:
                QMessageBox.warning(self, "Too few points", "Click more points along the curve (>= 10).")
                return
            self.mode = "idle"
            self.btn_compute.setEnabled(True)
            self.msg(f"Tracing finished: {len(self.curve_pts)} points. Click 'Compute f(t), f', f'', f'''.")
            self.canvas.setFocus()

    def px_to_data(self, px, py):
        # X: linear mapping from pixel-x to time, using two calibration clicks
        (px1, x1), (px2, x2) = self.cal_x
        ax = (x2 - x1) / (px2 - px1)
        bx = x1 - ax * px1
        t = ax * px + bx

        # Y: linear mapping from pixel-y to mV, using two calibration clicks
        # Note: pixel-y increases downward in images.
        (py1, y1), (py2, y2) = self.cal_y
        ay = (y2 - y1) / (py2 - py1)
        by = y1 - ay * py1
        mv = ay * py + by
        return t, mv

    def compute_all(self):
        try:
            if len(self.curve_pts) < 10:
                QMessageBox.warning(self, "No curve", "Trace curve points first.")
                return

            pts = np.array(self.curve_pts, dtype=float)
            px = pts[:, 0]
            py = pts[:, 1]

            t_raw, mv_raw = self.px_to_data(px, py)

            # Invert sign (image y grows downward). Estimate baseline with median.
            baseline = np.median(mv_raw)
            u_mV = baseline - mv_raw

            # Sort by time
            order = np.argsort(t_raw)
            t = t_raw[order]
            u = u_mV[order]

            # Drop duplicate t
            mask = np.concatenate([[True], np.diff(t) > 1e-12])
            t = t[mask]
            u = u[mask]

            self.t_sec = t
            self.u_mV = u

            # Spline interpolation + derivatives
            t_grid = np.linspace(t.min(), t.max(), 3000)
            s = 0.01 * len(t)  # smoothing factor (increase if derivatives too noisy)
            spl = UnivariateSpline(t, u, s=s)

            self.t_grid = t_grid
            self.f = spl(t_grid)
            self.f1 = spl.derivative(1)(t_grid)
            self.f2 = spl.derivative(2)(t_grid)
            self.f3 = spl.derivative(3)(t_grid)

            # Show f(t) in main canvas
            self.ax.clear()
            self.ax.plot(t, u, ".", markersize=3, label="Digitized points")
            self.ax.plot(t_grid, self.f, "-", label="f(t) spline")
            self.ax.set_xlabel("t (s)")
            self.ax.set_ylabel("U (mV)")
            self.ax.set_title("ECG: f(t)")
            self.ax.legend()
            self.canvas.draw()
            self.canvas.setFocus()

            # Prepare separate figures for export (f, f', f'', f''')
            self.fig_f = Figure(figsize=(10, 4))
            ax1 = self.fig_f.add_subplot(111)
            ax1.plot(t, u, ".", markersize=3)
            ax1.plot(t_grid, self.f, "-")
            ax1.set_xlabel("t (s)")
            ax1.set_ylabel("U (mV)")
            ax1.set_title("f(t)")

            self.fig_f1 = Figure(figsize=(10, 4))
            ax2 = self.fig_f1.add_subplot(111)
            ax2.plot(t_grid, self.f1)
            ax2.set_xlabel("t (s)")
            ax2.set_ylabel("dU/dt (mV/s)")
            ax2.set_title("f'(t)")

            self.fig_f2 = Figure(figsize=(10, 4))
            ax3 = self.fig_f2.add_subplot(111)
            ax3.plot(t_grid, self.f2)
            ax3.set_xlabel("t (s)")
            ax3.set_ylabel("d²U/dt²")
            ax3.set_title("f''(t)")

            self.fig_f3 = Figure(figsize=(10, 4))
            ax4 = self.fig_f3.add_subplot(111)
            ax4.plot(t_grid, self.f3)
            ax4.set_xlabel("t (s)")
            ax4.set_ylabel("d³U/dt³")
            ax4.set_title("f'''(t)")

            self.btn_export.setEnabled(True)
            self.msg("Computed f(t), f'(t), f''(t), f'''(t). You can export CSV + PNG plots now.")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def export_outputs(self):
        if self.t_sec is None:
            QMessageBox.warning(self, "Nothing to export", "Compute results first.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not out_dir:
            return

        # CSV
        df = pd.DataFrame({"t_sec": self.t_sec, "u_mV": self.u_mV})
        csv_path = f"{out_dir}/ecg_t_mV.csv"
        df.to_csv(csv_path, index=False)

        # PNG plots
        self.fig_f.savefig(f"{out_dir}/plot_f.png", dpi=200, bbox_inches="tight")
        self.fig_f1.savefig(f"{out_dir}/plot_f_prime.png", dpi=200, bbox_inches="tight")
        self.fig_f2.savefig(f"{out_dir}/plot_f_double_prime.png", dpi=200, bbox_inches="tight")
        self.fig_f3.savefig(f"{out_dir}/plot_f_triple_prime.png", dpi=200, bbox_inches="tight")

        QMessageBox.information(
            self,
            "Export done",
            f"Saved:\n{csv_path}\nplot_f.png\nplot_f_prime.png\nplot_f_double_prime.png\nplot_f_triple_prime.png"
        )

    def reset_all(self):
        self.img = None
        self.img_path = None
        self.cal_x = []
        self.cal_y = []
        self.curve_pts = []
        self.mode = "idle"

        self.t_sec = self.u_mV = self.t_grid = self.f = self.f1 = self.f2 = self.f3 = None
        self.fig_f = self.fig_f1 = self.fig_f2 = self.fig_f3 = None

        self.ax.clear()
        self.ax.set_title("Load image to start")
        self.ax.axis("off")
        self.canvas.draw()
        self.canvas.setFocus()

        self.btn_calx.setEnabled(False)
        self.btn_caly.setEnabled(False)
        self.btn_trace.setEnabled(False)
        self.btn_compute.setEnabled(False)
        self.btn_export.setEnabled(False)

        self.msg("Reset. Load image to start.")


def main():
    app = QApplication(sys.argv)
    w = ECGApp()
    w.resize(1200, 650)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()