"""Animated GIF of trajectory building up over time (no trail)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from dataset import SingleStrideWindowPaddedDataset

dataset = SingleStrideWindowPaddedDataset("M1182_PAG", 4, 108, is_train=False)

x, y = [], []
for i in range(len(dataset)):
    groups, pos, length, gathered_spikes, is_not_padding = dataset[i]
    if is_not_padding[0]:
        x.append(pos[0].item())
        y.append(pos[1].item())

x = np.asarray(x)
y = np.asarray(y)
N = len(x)

pad = 0.02 * max(x.max() - x.min(), y.max() - y.min())
xlim = (x.min() - pad, x.max() + pad)
ylim = (y.min() - pad, y.max() + pad)

target_frames = 220
step = max(1, N // target_frames)
indices = list(range(1, N + 1, step))
if indices[-1] != N:
    indices.append(N)

frames = []
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

for k in indices:
    ax.clear()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Trajectory (t = {k - 1}/{N - 1})")

    t = np.arange(k)
    colors = cmap(t / max(1, N - 1))

    ax.plot(x[:k], y[:k], color="black", alpha=0.15, linewidth=1)
    ax.scatter(x[:k], y[:k], c=colors, s=10, linewidths=0, alpha=0.95)
    ax.scatter(
        [x[k - 1]],
        [y[k - 1]],
        s=60,
        c="red",
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # RGB # type: ignore
    frames.append(frame.copy())

plt.close(fig)

imageio.mimsave("assets/trajectory.gif", frames, duration=0.05)
