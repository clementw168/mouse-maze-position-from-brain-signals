"""Mouse trajectory as a single line colored by timestep (LineCollection)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

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

points = np.column_stack([x, y])
segments = np.stack([points[:-1], points[1:]], axis=1)  # (N-1, 2, 2)

norm = Normalize(vmin=0, vmax=len(segments))
lc = LineCollection(segments, cmap="viridis", norm=norm)  # type: ignore
lc.set_array(np.arange(len(segments)))
lc.set_linewidth(2.0)
lc.set_alpha(0.95)

fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
ax.add_collection(lc)

# Optional: small markers for start/end
ax.scatter([x[0]], [y[0]], c="lime", s=60, edgecolor="black", linewidth=0.6, zorder=3)
ax.scatter([x[-1]], [y[-1]], c="red", s=60, edgecolor="black", linewidth=0.6, zorder=3)

# Set bounds so collection is fully visible
pad = 0.02 * max(x.max() - x.min(), y.max() - y.min())
ax.set_xlim(x.min() - pad, x.max() + pad)
ax.set_ylim(y.min() - pad, y.max() + pad)

cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Timestep")

ax.set_title(f"Mouse trajectory (colored line by time, N={len(x)})")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("assets/trajectory_visual.png", bbox_inches="tight")
plt.close()
