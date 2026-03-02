import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

# --- Load predictions (first 1600 points) ---
df = pd.read_csv("predictions.csv").iloc[:1600].reset_index(drop=True)

x_real = df["x_real"].to_numpy()
y_real = df["y_real"].to_numpy()
x_pred = df["x_pred"].to_numpy()
y_pred = df["y_pred"].to_numpy()

N = len(df)

mse = np.mean((x_real - x_pred) ** 2 + (y_real - y_pred) ** 2)
print("MSE:", mse)

# --- Fixed bounds (use both real+pred so everything fits) ---
x_all = np.concatenate([x_real, x_pred])
y_all = np.concatenate([y_real, y_pred])

pad = 0.02 * max(x_all.max() - x_all.min(), y_all.max() - y_all.min())
xlim = (x_all.min() - pad, x_all.max() + pad)
ylim = (y_all.min() - pad, y_all.max() + pad)

# --- Animation controls (same style as your example) ---
trail = 80
target_frames = 240
step = max(1, N // target_frames)
indices = list(range(0, N, step))
if indices[-1] != N - 1:
    indices.append(N - 1)

cmap = plt.get_cmap("viridis")
norm = Normalize(vmin=0, vmax=N - 1)

frames = []
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

# We'll create one colorbar on the first frame (from REAL scatter)
cbar = None

for i in indices:
    start = max(0, i - trail + 1)

    ax.clear()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Real vs Pred (t={i}/{N - 1}, trail={trail}) | MSE={mse:.4f}")

    ts = np.arange(start, i + 1)
    age_alpha = np.linspace(0.15, 1.0, len(ts))  # fade older -> newer

    # --- REAL trail ---
    xs_r = x_real[start : i + 1]
    ys_r = y_real[start : i + 1]
    ax.plot(xs_r, ys_r, color="tab:blue", alpha=0.18, linewidth=1)

    sc_real = ax.scatter(
        xs_r,
        ys_r,
        c=ts,
        cmap=cmap,
        norm=norm,
        s=22,
        alpha=age_alpha,  # type: ignore
        linewidths=0,
        label="real",
    )

    ax.scatter(
        [x_real[i]],
        [y_real[i]],
        s=90,
        c=[cmap(norm(i))],
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
    )

    # --- PRED trail (same colormap/time colors) ---
    xs_p = x_pred[start : i + 1]
    ys_p = y_pred[start : i + 1]
    ax.plot(xs_p, ys_p, color="tab:orange", alpha=0.18, linewidth=1)

    ax.scatter(
        xs_p,
        ys_p,
        c=ts,
        cmap=cmap,
        norm=norm,
        s=22,
        alpha=age_alpha,  # type: ignore
        linewidths=0,
        label="pred",
        marker="x",
    )

    ax.scatter(
        [x_pred[i]],
        [y_pred[i]],
        s=90,
        c=[cmap(norm(i))],
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
        marker="x",
    )

    # Connector for instantaneous error
    ax.plot(
        [x_real[i], x_pred[i]],
        [y_real[i], y_pred[i]],
        color="tab:red",
        alpha=0.5,
        linewidth=1,
        zorder=3,
    )

    ax.legend(loc="lower right")

    # Add colorbar once (static)
    if cbar is None:
        cbar = fig.colorbar(sc_real, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Timestep")

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # RGB # type: ignore
    frames.append(frame.copy())

plt.close(fig)

os.makedirs("visualization", exist_ok=True)
imageio.mimsave("assets/real_vs_pred_trail.gif", frames, duration=0.05)
print("Saved: assets/real_vs_pred_trail.gif")
