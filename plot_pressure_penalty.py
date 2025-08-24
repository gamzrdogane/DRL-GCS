import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------
# Pressure Penalty Parameters
# ----------------------------
P_INJECTION = 43.0     # MPa, green limit (I)
P_EXCEED    = 49.0     # MPa, red limit (E)
PRESSURE_K  = np.log(5)  # steepness of sigmoid
SCALE       = 1.0        # sigmoid scale factor
P_RANGE     = np.arange(38.0, 51.01, 0.1)  # full range for plotting

# ----------------------------
# Pressure Penalty Function
# ----------------------------
midpoint = 0.5 * (P_INJECTION + P_EXCEED)
penalty = 1.0 / (1.0 + np.exp(-PRESSURE_K * (P_RANGE - midpoint) / SCALE))

# Normalize penalty values for colormap
penalty_norm = (penalty - penalty.min()) / (penalty.max() - penalty.min())

# ----------------------------
# Build Colored Line Segments
# ----------------------------
cmap_gyr = LinearSegmentedColormap.from_list("gyr", ["green", "yellow", "red"])
colors = cmap_gyr(penalty_norm)

points   = np.array([P_RANGE, penalty]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
line     = LineCollection(segments, colors=colors[:-1], linewidths=5)

# ----------------------------
# Plotting
# ----------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.add_collection(line)
ax.set_xlim(P_RANGE.min(), P_RANGE.max())
ax.set_ylim(0, 1.05)

# Vertical dashed lines at I and E
ax.axvline(x=P_INJECTION,  color='k', linestyle='--')
ax.axvline(x=P_EXCEED,     color='k', linestyle='--')

# Horizontal dotted reference lines
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)

# Math-style labels
ax.set_xlabel(r'$P_{\max,\,t}$ [MPa]', fontsize=20)
ax.set_ylabel(r'$P_{\mathrm{pen},\,t}$', fontsize=20)

# Y ticks at penalty levels
ax.set_yticks([0.5, 1.0])
ax.tick_params(axis='y', labelsize=16, length=0)
ax.tick_params(axis='x', length=0)
ax.set_xticks([])

# Place colored math labels for I and E
ax.text(P_INJECTION, -0.03, r'I', transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=20)
ax.text(P_EXCEED,    -0.03, r'E', transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=20)

# Save high-resolution figure
plt.tight_layout()
plt.show()
