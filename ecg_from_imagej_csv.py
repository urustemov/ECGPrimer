import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# =========================
# НАСТРОЙКИ
# =========================
CSV_PATH = "ecg_points_mm.csv"   # <-- поставь своё имя файла

PAPER_SPEED_MM_PER_S = 25.0      # 25 мм/с
GAIN_MM_PER_MV = 5.0             # 5 мм/мВ

# Как выбрать baseline:
# 1) Авто: возьмем медиану Y (часто близко к изолинии на длинном участке)
# 2) Ручной режим: введи число Y_BASELINE_MM вручную
USE_AUTO_BASELINE = True
Y_BASELINE_MM = None  # например 86.0, если решишь вручную

# Сглаживание сплайна: чем больше, тем глаже (меньше шум в производных)
SMOOTHING_FACTOR = 0.002

# =========================
# ЗАГРУЗКА CSV
# =========================
df = pd.read_csv(CSV_PATH)

# Подхватим нужные столбцы: в ImageJ чаще всего есть X и Y
# Если у тебя названия отличаются (например "X (mm)"), просто напиши мне — скажу что заменить.
if "X" not in df.columns or "Y" not in df.columns:
    raise ValueError(f"Не вижу столбцы X и Y. Есть: {list(df.columns)}")

x_mm = df["X"].to_numpy(dtype=float)
y_mm = df["Y"].to_numpy(dtype=float)

# =========================
# ПЕРЕВОД мм -> секунды и мВ
# =========================
t_sec = x_mm / PAPER_SPEED_MM_PER_S

if USE_AUTO_BASELINE:
    # медиана обычно хорошо работает как оценка "середины" кривой
    y0 = np.median(y_mm)
else:
    if Y_BASELINE_MM is None:
        raise ValueError("Задай Y_BASELINE_MM или включи USE_AUTO_BASELINE=True")
    y0 = float(Y_BASELINE_MM)

# Важно: у изображений Y растет вниз, а в ЭКГ вверх = +
u_mV = (y0 - y_mm) / GAIN_MM_PER_MV

# Сортировка по времени
order = np.argsort(t_sec)
t_sec = t_sec[order]
u_mV = u_mV[order]

# Удалим повторяющиеся t, если есть
mask = np.concatenate([[True], np.diff(t_sec) > 1e-12])
t_sec = t_sec[mask]
u_mV = u_mV[mask]

out = pd.DataFrame({"t_sec": t_sec, "u_mV": u_mV})
out.to_csv("ecg_t_mV.csv", index=False)
print("✅ Сохранено: ecg_t_mV.csv")
print("baseline (Y0) =", y0, "mm")

# =========================
# ИНТЕРПОЛЯЦИЯ: f(t)
# =========================
# Сплайн со сглаживанием, чтобы производные не были "пилой"
s = SMOOTHING_FACTOR * len(t_sec)
spl = UnivariateSpline(t_sec, u_mV, s=s)

t_grid = np.linspace(t_sec.min(), t_sec.max(), 3000)
f = spl(t_grid)
f1 = spl.derivative(1)(t_grid)
f2 = spl.derivative(2)(t_grid)
f3 = spl.derivative(3)(t_grid)

# =========================
# ГРАФИКИ
# =========================
plt.figure(figsize=(14,5))
plt.plot(t_sec, u_mV, ".", markersize=3, label="Точки (из ImageJ)")
plt.plot(t_grid, f, "-", label="f(t) (сплайн)")
plt.xlabel("t, сек")
plt.ylabel("U, мВ")
plt.title("ECG: f(t)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,5))
plt.plot(t_grid, f1)
plt.xlabel("t, сек")
plt.ylabel("dU/dt, мВ/с")
plt.title("ECG: f'(t) — наклон")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,5))
plt.plot(t_grid, f2)
plt.xlabel("t, сек")
plt.ylabel("d²U/dt²")
plt.title("ECG: f''(t) — 2-я производная (кривизна в простом виде)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,5))
plt.plot(t_grid, f3)
plt.xlabel("t, сек")
plt.ylabel("d³U/dt³")
plt.title("ECG: f'''(t) — 3-я производная")
plt.tight_layout()
plt.show()