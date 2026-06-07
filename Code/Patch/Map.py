"""
Visualize values as color-coded dot plots
Author: Isaac Chang
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def stats(x, y, bins):
    x = np.array(x)
    y = np.array(y)

    binidx = np.digitize(x, bins) - 1

    means = []
    sems = []
    vals = []

    for i in range(len(bins) - 1):
        val = y[binidx == i]
        vals.append(val)
        if len(val) > 0:
            means.append(np.mean(val))
            sems.append(np.std(val) / np.sqrt(len(val)))
        else:
            means.append(np.nan)
            sems.append(np.nan)
    centers = (bins[:-1] + bins[1:]) / 2

    return centers, np.array(means), np.array(sems), vals


path_in = r"C:\Users\isaac.chang\Desktop\Data\Patch\KCl_Opto\Map"
path_out = path_in
pathinfo = path_in

dotInfo = pd.read_csv(os.path.join(pathinfo, "info.csv"))

val = dotInfo["ECl"].tolist()
ap = dotInfo["AP"].tolist()
ml = dotInfo["ML"].tolist()
dv = dotInfo["DV"].tolist()
val = np.array(val)
ap = np.array(ap)
ml = np.array(ml)
dv = np.array(dv)

vmin = -100
vmax = -20
offset = 1

cmap = cm.get_cmap("viridis")
cmapscale = cmap(np.linspace(0, 1, 256))
newcmap = LinearSegmentedColormap.from_list('trunc', cmapscale)

for i in np.unique(dv):
    mask = dv == i
    scatter = plt.scatter(ap[mask] + (i - 1) * offset, ml[mask],
                          c=val[mask], cmap=newcmap, vmin=vmin, vmax=vmax)

plt.colorbar(scatter)
plt.ylim(1, 4.7)
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.savefig(os.path.join(path_out, " Graph1.png"), format="png")
plt.close()

apbins = np.arange(-0.4, -2.2, -0.2)
mlbins = np.arange(1.0, 3.2, 0.2)
dvbins = np.arange(0.5, 6.0, 1.0)

apC, apMean, apSem, apVals = stats(ap, val, apbins)
mlC, mlMean, mlSem, mlVals = stats(ml, val, mlbins)
dvC, dvMean, dvSem, dvVals = stats(dv, val, dvbins)

fig, axes = plt.subplots(3, 1)
fig.set_figheight(10)
fig.set_figwidth(5)

axes[0].scatter(ap, val, alpha=0.3)
axes[0].plot(apC, apMean)
axes[0].fill_between(apC, apMean-apSem, apMean+apSem, alpha=0.3)
axes[0].set_xlim(apbins[0], apbins[-1])
axes[0].set_xlabel("AP")

axes[1].scatter(ml, val, alpha=0.3)
axes[1].plot(mlC, mlMean)
axes[1].fill_between(mlC, mlMean-mlSem, mlMean+mlSem, alpha=0.3)
axes[1].set_xlim(mlbins[0], mlbins[-1])
axes[1].set_xlabel("ML")

axes[2].scatter(dv, val, alpha=0.3,)
axes[2].plot(dvC, dvMean)
axes[2].fill_between(dvC, dvMean-dvSem, dvMean+dvSem, alpha=0.3)
axes[2].set_xlim(0.5, 5.5)
axes[2].set_xlabel("DV")

apdf = pd.DataFrame({
    "center": np.repeat(apC, [len(v) for v in apVals]),
    "value": np.concatenate(apVals)
})

mldf = pd.DataFrame({
    "center": np.repeat(mlC, [len(v) for v in mlVals]),
    "value": np.concatenate(mlVals)
})

dvdf = pd.DataFrame({
    "center": np.repeat(dvC, [len(v) for v in dvVals]),
    "value": np.concatenate(dvVals)
})

df = pd.concat([apdf, mldf, dvdf], ignore_index=True)

plt.savefig(os.path.join(path_out, " Graph2.png"), format="png")
plt.close()

fileName = "Map_Results.csv"
filePath = os.path.join(path_out, fileName)
df.to_csv(filePath)
