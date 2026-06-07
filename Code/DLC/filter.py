"""
Filter DLC files. First step of preprocessing.
Author: Isaac Chang
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os
import pandas as pd
import scipy
from scipy.signal import medfilt


def load(path_in):
    files = []
    for i in os.listdir(path_in):
        if os.path.splitext(i)[-1].lower() == ".csv":
            files.append(i)
    files.sort()
    return files


def init():
    for pt in points:
        pt.set_data([], [])
    return points

def update(frame):
    for i, bp in enumerate(bodyparts):
        x = df[(scorer, bp, "x")].values[frame]
        y = df[(scorer, bp, "y")].values[frame]
        points[i].set_data(x, y)
    return points


path_in = r"\\hive\Paz-Lab\Isaac\DLC\videos\opto\Npas1-eNpHR-sustained"
path_out = path_in

files = load(path_in)

print("Analyzing " + str(len(files)) + " files below:")
print(files)

P_TH = 0.3
MAX_PX_PER_FRAME = 30
MEDFILT_KERNEL = 5
SPIKE_NEIGHBOR_THRESH = 3
SPIKE_OUTLIER_THRESH = 6 
N_FRAMES = 1000
FPS = 10

for index, value in enumerate(files):
    df = pd.read_csv(os.path.join(path_in, value),
                     header=[0, 1, 2], index_col=0)

    scorer = df.columns.levels[0][0]
    bodyparts = df.columns.levels[1]

    for bp in bodyparts:
        x = df[(scorer, bp, "x")].values
        y = df[(scorer, bp, "y")].values
        p = df[(scorer, bp, "likelihood")].values

        x[p < P_TH] = np.nan
        y[p < P_TH] = np.nan

        valid = ~np.isnan(x)

        idx_valid = np.where(valid)[0]
        if len(idx_valid) < 2:
            continue
        for k in range(len(idx_valid) - 1):
            i0 = idx_valid[k]
            i1 = idx_valid[k + 1]
            gap = i1 - i0
            if gap > 1:
                for j in range(1, gap):
                    frac = j / gap
                    x[i0 + j] = x[i0] + frac * (x[i1] - x[i0])
                    y[i0 + j] = y[i0] + frac * (y[i1] - y[i0])

        first = idx_valid[0]
        x[:first] = x[first]
        y[:first] = y[first]
        last = idx_valid[-1]
        x[last + 1:] = x[last]
        y[last + 1:] = y[last]

        for i in range(1, len(x) - 1):
            d_prev_next = np.sqrt(
                (x[i-1] - x[i+1])**2 + (y[i-1] - y[i+1])**2
            )
            d_prev = np.sqrt(
                (x[i] - x[i-1])**2 + (y[i] - y[i-1])**2
            )
            d_next = np.sqrt(
                (x[i] - x[i+1])**2 + (y[i] - y[i+1])**2
            )

            if (
                d_prev_next < SPIKE_NEIGHBOR_THRESH and
                d_prev > SPIKE_OUTLIER_THRESH and
                d_next > SPIKE_OUTLIER_THRESH
            ):
                x[i] = 0.5 * (x[i-1] + x[i+1])
                y[i] = 0.5 * (y[i-1] + y[i+1])

        for i in range(1, len(x)):
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist > MAX_PX_PER_FRAME:
                scale = MAX_PX_PER_FRAME / dist
                x[i] = x[i-1] + dx * scale
                y[i] = y[i-1] + dy * scale

        x = medfilt(x, kernel_size=MEDFILT_KERNEL)
        y = medfilt(y, kernel_size=MEDFILT_KERNEL)

        df[(scorer, bp, "x")] = x
        df[(scorer, bp, "y")] = y

    outname = value.replace(".csv", "_filtered.h5")
    df.to_hdf(os.path.join(path_out, outname), key="df", mode="w")

    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")

    bodyparts = list(bodyparts)
    colors = plt.cm.tab10(np.linspace(0, 1, len(bodyparts)))

    xs = []
    ys = []
    for bp in bodyparts:
        xs.append(df[(scorer, bp, "x")].values[:N_FRAMES])
        ys.append(df[(scorer, bp, "y")].values[:N_FRAMES])

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(np.nanmin(xs), np.nanmax(xs))
    ax.set_ylim(np.nanmax(ys), np.nanmin(ys))

    points = []
    for bp, c in zip(bodyparts, colors):
        pt, = ax.plot([], [], "o", color=c, label=bp, markersize=6)
        points.append(pt)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    anim = FuncAnimation(
        fig,
        update,
        frames=N_FRAMES,
        init_func=init,
        blit=True
    )
    plt.show()