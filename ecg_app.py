import os
import sys
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QPushButton, QLabel, QTextEdit, QVBoxLayout, QHBoxLayout,
    QInputDialog
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from scipy.interpolate import UnivariateSpline


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


class ECGDigitizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "ECG Digitizer + Normalized Comparison: "
            "f(t), f'(t), f''(t), f'''(t), f''''(t), f'''''(t)"
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
        self.f = None
        self.f1 = None
        self.f2 = None
        self.f3 = None
        self.f4 = None
        self.f5 = None

        # export figures
        self.fig_f = None
        self.fig_f1 = None
        self.fig_f2 = None
        self.fig_f3 = None
        self.fig_f4 = None
        self.fig_f5 = None

        self.fig_fn = None
        self.fig_f1n = None
        self.fig_f2n = None
        self.fig_f3n = None
        self.fig_f4n = None
        self.fig_f5n = None

        # comparison data
        self.compare_df = None

        self.setup_ui()
        self.write_log(
            "Шаги:\n"
            "1) Load ECG Image\n"
            "2) Calibrate X -> кликни 2 точки по оси/сетке времени и введи их значения\n"
            "3) Calibrate Y -> кликни 2 точки по оси/сетке амплитуды и введи их значения\n"
            "4) Trace Curve Points -> поставь точки по кривой\n"
            "5) Compute -> считает f, f', f'', f''', f'''', f'''''\n"
            "6) Export CSV + PNG\n"
            "7) Для сравнения двух снимков: обработай первый, экспортируй CSV; "
            "потом обработай второй и нажми Compare with Exported CSV\n"
        )

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)

        left = QVBoxLayout()
        right = QVBoxLayout()

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

        self.btn_compute = QPushButton("Compute f(t) ... f'''''(t)")
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

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_all)
        left.addWidget(self.btn_reset)

        left.addWidget(QLabel("Log"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        left.addWidget(self.log_box, 1)

        self.canvas = ECGCanvas(parent_window=self)
        right.addWidget(self.canvas)

        root.addLayout(left, 0)
        root.addLayout(right, 1)

    def write_log(self, text: str):
        self.log_box.append(text)

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
        ax = self.canvas.ax
        ax.clear()

        if self.img is not None:
            ax.imshow(self.img)
            ax.set_title("ECG image")
        else:
            ax.set_title("No image loaded")

        # calibration points
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

    def compute_all(self):
        try:
            if len(self.curve_pts) < 6:
                QMessageBox.warning(self, "Warning", "Trace at least 6 points first.")
                return

            pts = np.array(self.curve_pts, dtype=float)

            t = np.array([self.pixel_to_time(x) for x, _ in pts], dtype=float)
            u = np.array([self.pixel_to_voltage(y) for _, y in pts], dtype=float)

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

            if len(t) < 6:
                QMessageBox.warning(self, "Warning", "Need at least 6 unique time points.")
                return

            self.t_sec = t
            self.u_mV = u

            t_grid = np.linspace(t.min(), t.max(), 1000)

            # higher smoothing if needed
            s_factor = 0.01
            s = s_factor * len(t)

            # quintic spline for up to 5th derivative
            spl = UnivariateSpline(t, u, s=s, k=5)

            self.t_grid = t_grid
            self.f = spl(t_grid)
            self.f1 = spl.derivative(1)(t_grid)
            self.f2 = spl.derivative(2)(t_grid)
            self.f3 = spl.derivative(3)(t_grid)
            self.f4 = spl.derivative(4)(t_grid)
            self.f5 = spl.derivative(5)(t_grid)

            # normalized x coordinate from 0 to 1
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
                "f(t), f'(t), f''(t), f'''(t), f''''(t), f'''''(t)\n"
                "Normalized coordinate x_norm in [0, 1] created for comparison."
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Computation failed:\n{e}")

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
        self.fig_f4 = self.build_single_figure(
            self.t_grid, self.f4, "f''''(t)", "t (s)", "d⁴U/dt⁴"
        )
        self.fig_f5 = self.build_single_figure(
            self.t_grid, self.f5, "f'''''(t)", "t (s)", "d⁵U/dt⁵"
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
        self.fig_f4n = self.build_single_figure(
            self.x_norm, self.f4, "f''''(x_norm)", "x_norm", "d⁴U/dt⁴"
        )
        self.fig_f5n = self.build_single_figure(
            self.x_norm, self.f5, "f'''''(x_norm)", "x_norm", "d⁵U/dt⁵"
        )

    def plot_absolute_main(self):
        if self.t_grid is None or self.f is None:
            QMessageBox.warning(self, "Warning", "Compute data first.")
            return

        ax = self.canvas.ax
        ax.clear()
        ax.plot(self.t_grid, self.f, label="f(t)")
        ax.set_title("Absolute coordinates")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("U")
        ax.grid(True)
        ax.legend()
        self.canvas.draw()

    def plot_normalized_main(self):
        if self.x_norm is None or self.f is None:
            QMessageBox.warning(self, "Warning", "Compute data first.")
            return

        ax = self.canvas.ax
        ax.clear()
        ax.plot(self.x_norm, self.f, label="f(x_norm)")
        ax.plot(self.x_norm, self.f1, label="f'(x_norm)")
        ax.plot(self.x_norm, self.f2, label="f''(x_norm)")
        ax.set_title("Normalized coordinates for comparison")
        ax.set_xlabel("x_norm (0 = start, 1 = end)")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()
        self.canvas.draw()

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

            csv_path = os.path.join(out_dir, f"{base_name}.csv")

            df = pd.DataFrame({
                "t_sec": self.t_grid,
                "x_norm": self.x_norm,
                "f": self.f,
                "f1": self.f1,
                "f2": self.f2,
                "f3": self.f3,
                "f4": self.f4,
                "f5": self.f5
            })
            df.to_csv(csv_path, index=False)

            self.fig_f.savefig(os.path.join(out_dir, f"{base_name}_f.png"), dpi=200, bbox_inches="tight")
            self.fig_f1.savefig(os.path.join(out_dir, f"{base_name}_f1.png"), dpi=200, bbox_inches="tight")
            self.fig_f2.savefig(os.path.join(out_dir, f"{base_name}_f2.png"), dpi=200, bbox_inches="tight")
            self.fig_f3.savefig(os.path.join(out_dir, f"{base_name}_f3.png"), dpi=200, bbox_inches="tight")
            self.fig_f4.savefig(os.path.join(out_dir, f"{base_name}_f4.png"), dpi=200, bbox_inches="tight")
            self.fig_f5.savefig(os.path.join(out_dir, f"{base_name}_f5.png"), dpi=200, bbox_inches="tight")

            self.fig_fn.savefig(os.path.join(out_dir, f"{base_name}_norm_f.png"), dpi=200, bbox_inches="tight")
            self.fig_f1n.savefig(os.path.join(out_dir, f"{base_name}_norm_f1.png"), dpi=200, bbox_inches="tight")
            self.fig_f2n.savefig(os.path.join(out_dir, f"{base_name}_norm_f2.png"), dpi=200, bbox_inches="tight")
            self.fig_f3n.savefig(os.path.join(out_dir, f"{base_name}_norm_f3.png"), dpi=200, bbox_inches="tight")
            self.fig_f4n.savefig(os.path.join(out_dir, f"{base_name}_norm_f4.png"), dpi=200, bbox_inches="tight")
            self.fig_f5n.savefig(os.path.join(out_dir, f"{base_name}_norm_f5.png"), dpi=200, bbox_inches="tight")

            QMessageBox.information(
                self,
                "Export done",
                f"Saved:\n{csv_path}\n\n"
                "Absolute and normalized PNG graphs were also saved."
            )
            self.write_log(f"Exported CSV and PNG files to: {out_dir}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{e}")

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

            required_cols = {"x_norm", "f", "f1", "f2", "f3", "f4", "f5"}
            if not required_cols.issubset(set(df2.columns)):
                raise ValueError(
                    "CSV does not contain required columns: "
                    "x_norm, f, f1, f2, f3, f4, f5"
                )

            self.compare_df = df2.copy()

            common_x = np.linspace(0.0, 1.0, 400)

            cur_f = np.interp(common_x, self.x_norm, self.f)
            cur_f1 = np.interp(common_x, self.x_norm, self.f1)
            cur_f2 = np.interp(common_x, self.x_norm, self.f2)
            cur_f3 = np.interp(common_x, self.x_norm, self.f3)
            cur_f4 = np.interp(common_x, self.x_norm, self.f4)
            cur_f5 = np.interp(common_x, self.x_norm, self.f5)

            ref_x = df2["x_norm"].to_numpy(dtype=float)
            ref_f = np.interp(common_x, ref_x, df2["f"].to_numpy(dtype=float))
            ref_f1 = np.interp(common_x, ref_x, df2["f1"].to_numpy(dtype=float))
            ref_f2 = np.interp(common_x, ref_x, df2["f2"].to_numpy(dtype=float))
            ref_f3 = np.interp(common_x, ref_x, df2["f3"].to_numpy(dtype=float))
            ref_f4 = np.interp(common_x, ref_x, df2["f4"].to_numpy(dtype=float))
            ref_f5 = np.interp(common_x, ref_x, df2["f5"].to_numpy(dtype=float))

            fig = Figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            ax.plot(common_x, cur_f2, label="Current f''")
            ax.plot(common_x, ref_f2, label="Reference f''")
            ax.plot(common_x, cur_f, label="Current f", alpha=0.6)
            ax.plot(common_x, ref_f, label="Reference f", alpha=0.6)

            ax.set_title("Comparison on same normalized coordinates")
            ax.set_xlabel("x_norm (0 ... 1)")
            ax.set_ylabel("Value")
            ax.grid(True)
            ax.legend()

            self.canvas.fig.clear()
            self.canvas.fig.set_size_inches(12, 8)
            self.canvas.ax = self.canvas.fig.add_subplot(111)

            self.canvas.ax.plot(common_x, cur_f, label="Current f")
            self.canvas.ax.plot(common_x, ref_f, label="Reference f")
            self.canvas.ax.plot(common_x, cur_f1, label="Current f'")
            self.canvas.ax.plot(common_x, ref_f1, label="Reference f'")
            self.canvas.ax.plot(common_x, cur_f2, label="Current f''")
            self.canvas.ax.plot(common_x, ref_f2, label="Reference f''")
            self.canvas.ax.set_title("Comparison on same normalized coordinates")
            self.canvas.ax.set_xlabel("x_norm (0 = start, 1 = end)")
            self.canvas.ax.set_ylabel("Value")
            self.canvas.ax.grid(True)
            self.canvas.ax.legend()
            self.canvas.draw()

            self.write_log(
                f"Comparison loaded from: {csv_path}\n"
                "Both signals were resampled to the same normalized grid [0, 1]."
            )

            # quick numeric summary
            diff_f = np.max(np.abs(cur_f - ref_f))
            diff_f1 = np.max(np.abs(cur_f1 - ref_f1))
            diff_f2 = np.max(np.abs(cur_f2 - ref_f2))
            diff_f3 = np.max(np.abs(cur_f3 - ref_f3))
            diff_f4 = np.max(np.abs(cur_f4 - ref_f4))
            diff_f5 = np.max(np.abs(cur_f5 - ref_f5))

            self.write_log(
                "Max absolute differences on common normalized grid:\n"
                f"f   : {diff_f:.6f}\n"
                f"f'  : {diff_f1:.6f}\n"
                f"f'' : {diff_f2:.6f}\n"
                f"f''' : {diff_f3:.6f}\n"
                f"f'''' : {diff_f4:.6f}\n"
                f"f''''' : {diff_f5:.6f}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Comparison failed:\n{e}")

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
        self.f = None
        self.f1 = None
        self.f2 = None
        self.f3 = None
        self.f4 = None
        self.f5 = None

        self.fig_f = None
        self.fig_f1 = None
        self.fig_f2 = None
        self.fig_f3 = None
        self.fig_f4 = None
        self.fig_f5 = None

        self.fig_fn = None
        self.fig_f1n = None
        self.fig_f2n = None
        self.fig_f3n = None
        self.fig_f4n = None
        self.fig_f5n = None

        self.compare_df = None

        self.log_box.clear()
        self.write_log("Reset done.")
        self.redraw_main_canvas()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ECGDigitizer()
    win.show()
    sys.exit(app.exec())