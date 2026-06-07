"""
Process DLC filtered data files for opto population data
Author: Isaac Chang
"""

import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd


def load():
    files = []
    for i in os.listdir(pathfiles):
        if os.path.splitext(i)[-1].lower() == ".csv":
            files.append(i)
    files.sort()
    return files


def speedAnalyze(data):
    optoData = np.empty((optoTrial, lengthTrial))
    preSpeed = np.empty(optoTrial)
    stimSpeed = np.empty(optoTrial)
    speedChange = np.empty(optoTrial)

    for i in np.arange(optoTrial):
        start = int(firstoptoFrames[index] + i * lengthTrial)
        end = int(firstoptoFrames[index] + (i + 1) * lengthTrial)
        optoData[i] = data[start:end]
        allSpeed[index * optoTrial + i] = optoData[i]
        preSpeed[i] = np.mean(optoData[i][0:100])
        stimSpeed[i] = np.mean(optoData[i][100:200])
        speedChange[i] = stimSpeed[i] / preSpeed[i]
        axes[0, 1].plot(
            preSpeed[i], stimSpeed[i], "o", markersize=2, color="blue"
        )
        axes[1, 1].plot(
            preSpeed[i], speedChange[i], "o", markersize=2, color="blue"
        )
        axes[2, 1].plot(
            preSpeed[i], speedChange[i], "o", markersize=2, color="blue"
        )

    allpreSpeed.append(preSpeed.flatten())
    allSpeedChange.append(speedChange.flatten())

    return


def distanceAnalyze(dataX, dataY):

    preDistance = np.empty(optoTrial)
    stimDistance = np.empty(optoTrial)
    distanceChange = np.empty(optoTrial)

    XCenter = (np.max(dataX)+np.min(dataX))/2
    YCenter = (np.max(dataY)+np.min(dataY))/2

    for i in np.arange(optoTrial):
        start = int(firstoptoFrames[index] + i * lengthTrial)
        end = int(firstoptoFrames[index] + (i + 1) * lengthTrial)
        distance = np.empty(lengthTrial)
        XCor = np.empty(lengthTrial)
        YCor = np.empty(lengthTrial)

        for j, (x, y) in enumerate(zip(dataX[start:end], dataY[start:end])):
            distance[j] = ((x - XCenter)**2 + (y - YCenter)**2) ** 0.5
            XCor[j] = x
            YCor[j] = y

        preDistance[i] = np.mean(distance[0:100])
        stimDistance[i] = np.mean(distance[100:200])
        distanceChange[i] = stimDistance[i] - preDistance[i]
        allDistance[index * optoTrial + i] = distance
        allXCor[index * optoTrial + i] = XCor
        allYCor[index * optoTrial + i] = YCor
        axes2[0, 1].plot(
            preDistance[i], stimDistance[i], "o", markersize=2, color="blue"
        )
        axes2[1, 1].plot(
            preDistance[i], distanceChange[i], "o", markersize=2, color="blue"
        )

    allpreDistance.append(preDistance.flatten())
    allDistanceChange.append(distanceChange.flatten())

    return


optoTrial = 20
lengthTrial = 11183 // (optoTrial-1)
graphFormat = "png"
pathfiles = r"\\hive\Paz-Lab\Isaac\GPe\Data\Opto\Analysis\Npas1-eYFP-20Hz"
pathinfo = pathfiles
videoInfo = pd.read_csv(os.path.join(pathinfo, "videoInfo.csv"))
files = load()
files = [f for f in files if f[:-4] in set(videoInfo["Video file"])]
firstoptoFrames = videoInfo["First opto frame"].tolist()
lastoptoFrames = videoInfo["Last opto frame"].tolist()

outdf = pd.DataFrame(index=range(300))

print("Analyzing " + str(len(files)) + " files below:")
print(files)

allSpeed = np.empty((len(files)*optoTrial, lengthTrial))
allDistance = np.empty((len(files)*optoTrial, lengthTrial))
allXCor = np.empty((len(files)*optoTrial, lengthTrial))
allYCor = np.empty((len(files)*optoTrial, lengthTrial))
allpreSpeed = []
allSpeedChange = []
allpreDistance = []
allDistanceChange = []

fig, axes = plt.subplots(
    3, 3, figsize=(18, 12), gridspec_kw={"width_ratios": [3, 1, 1]}
)

fig2, axes2 = plt.subplots(
    3, 3, figsize=(18, 12), gridspec_kw={"width_ratios": [3, 1, 1]}
)

fig3, axes3 = plt.subplots(
    1, 1, figsize=(10, 10))

fig4, axes4 = plt.subplots(
    1, 1, figsize=(10, 10))

fig5, axes5 = plt.subplots(
    1, 6, figsize=(18, 3))

for index, value in enumerate(files):
    df = pd.read_csv(os.path.join(pathfiles, value))
    speed = df["speed"].to_numpy()
    XCor = df["XCor"].to_numpy()
    YCor = df["YCor"].to_numpy()
    speedAnalyze(speed)
    distanceAnalyze(XCor, YCor)
    print(index)

allpreSpeed = np.array(allpreSpeed).flatten()
allSpeedChange = np.array(allSpeedChange).flatten()
avgSpeedX = np.mean(allSpeed, axis=0)
semSpeedX = np.std(allSpeed, axis=0, ddof=1) / np.sqrt(len(allSpeed))

axes[0, 0].plot(avgSpeedX)
axes[0, 0].fill_between(
    np.arange(0, lengthTrial),
    avgSpeedX - semSpeedX,
    avgSpeedX + semSpeedX,
    alpha=0.3,
)

bins = np.logspace(-0.3, 0.5, 5)
bins = np.concatenate(([0], bins, [np.inf]))
binID = np.digitize(allpreSpeed, bins)

# cmap = cm.get_cmap('Greys')
cmap = cm.get_cmap('Blues')
colors = cmap(np.linspace(0.5, 1, len(bins)-1))[::-1]
norm = mcolors.Normalize(vmin=10**(-0.3), vmax=10**(0.5))
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[2, 2])

for i in range(1, len(bins)):
    mask = binID == i

    meanTrace = np.nanmean(allSpeed[mask], axis=0)
    semTrace = np.nanstd(allSpeed[mask], axis=0) / np.sqrt(np.sum(mask))

    phases = [0, 100, 200, 300, 400, 500, len(meanTrace)]

    meanTracePhases = np.array([
        np.nanmean(meanTrace[phases[i]:phases[i+1]])
        for i in range(len(phases)-1)
    ])

    semTracePhases = np.array([
        np.sqrt(
            np.nanmean(
                semTrace[phases[i]:phases[i+1]]**2
            )
        )
        for i in range(len(phases)-1)
    ])

    for j in allSpeed[mask]:
        axes[1, 0].plot(j, color=colors[i-1], alpha=0.05)
        axes[0, 2].plot([0, 1, 2, 3, 4, 5], [np.mean(j[0:100]), np.mean(j[100:200]), np.mean(j[200:300]),
                                             np.mean(j[300:400]), np.mean(j[400:500]), np.mean(j[500:])],
                        linestyle='-', color=colors[i-1], alpha=0.3)

    axes[1, 2].plot(
        np.arange(len(meanTracePhases)),
        meanTracePhases,
        color=colors[i-1],
        alpha=0.7
    )

    axes[1, 2].fill_between(
        np.arange(len(meanTracePhases)),
        meanTracePhases - semTracePhases,
        meanTracePhases + semTracePhases,
        color=colors[i-1],
        alpha=0.2
    )

    axes[2, 0].plot(
        np.arange(len(meanTrace)),
        meanTrace,
        color=colors[i-1],
        alpha=0.7
    )

    axes[2, 0].fill_between(
        np.arange(len(meanTrace)),
        meanTrace - semTrace,
        meanTrace + semTrace,
        color=colors[i-1],
        alpha=0.2
    )

    # outdf["Mean speed " + str(i)] = pd.Series(meanTracePhases)
    # outdf["SEM speed " + str(i)] = pd.Series(semTracePhases)

axes[0, 0].set_ylabel("Speed (cm/s)")
axes[0, 0].axvspan(100, 200, color="orange", alpha=0.2, zorder=0)
axes[0, 0].xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: int(x/10)))
axes[1, 0].set_ylabel("Speed (cm/s)")
axes[1, 0].axvspan(100, 200, color="orange", alpha=0.2, zorder=0)
axes[1, 0].set_ylim(0, 20)
axes[1, 0].xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: int(x/10)))
axes[2, 0].set_ylabel("Speed (cm/s)")
axes[2, 0].axvspan(100, 200, color="orange", alpha=0.2, zorder=0)
axes[2, 0].set_ylim(0.1, 10)
axes[2, 0].set_yscale('log')
axes[2, 0].xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: int(x/10)))
axes[2, 0].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Stim speed (cm/s)")
axes[0, 1].set_ylim(0, 10)
axes[0, 1].set_xlabel("Pre speed (cm/s)")
axes[0, 1].set_xlim(0, 10)
axes[0, 1].plot([0, 10], [0, 10], "--", color="black")
axes[1, 1].set_ylabel("Fold change")
axes[1, 1].set_ylim(0, 5)
axes[1, 1].set_xlabel("Pre speed (cm/s)")
axes[1, 1].set_xlim(0, 10)
axes[1, 1].axhline(1, ls=":")
axes[2, 1].set_ylabel("Fold change")
axes[2, 1].set_ylim(0.01, 100)
axes[2, 1].set_yscale('log')
axes[2, 1].set_xlabel("Pre speed (cm/s)")
axes[2, 1].set_xlim(0, 10)
axes[2, 1].axhline(1, ls=":")
axes[0, 2].set_ylabel("Speed (cm/s)")
axes[0, 2].set_yscale("log")
axes[0, 2].set_ylim(0.1, 10)
axes[0, 2].axvspan(0.5, 1.5, color="orange", alpha=0.2, zorder=0)
axes[1, 2].set_ylabel("Speed (cm/s)")
axes[1, 2].set_yscale('log')
axes[1, 2].set_ylim(0.1, 10)
axes[1, 2].axvspan(0.5, 1.5, color="orange", alpha=0.2, zorder=0)

if graphFormat == "svg":
    axes[1, 0].remove()
fig.savefig(os.path.join(
    pathfiles, f"Graph.{graphFormat}"), format=graphFormat)

allpreDistance = np.array(allpreDistance).flatten()
allDistanceChange = np.array(allDistanceChange).flatten()
avgDistance = np.mean(allDistance, axis=0)
semDistance = np.std(allDistance, axis=0, ddof=1) / np.sqrt(len(allDistance))

axes2[0, 0].plot(avgDistance)
axes2[0, 0].fill_between(
    np.arange(0, lengthTrial),
    avgDistance - semDistance,
    avgDistance + semDistance,
    alpha=0.3,
)

bins = np.linspace(7, 14, 8)
bins = np.concatenate(([0], bins, [np.inf]))
binID = np.digitize(allpreDistance, bins)

colors = cmap(np.linspace(0.5, 1, len(bins)-1))[::-1]
norm = mcolors.Normalize(vmin=7, vmax=15)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes2[2, 2])


for i in range(1, len(bins)):
    mask = binID == i

    meanTrace = np.nanmean(allDistance[mask], axis=0)
    semTrace = np.nanstd(allDistance[mask], axis=0) / np.sqrt(np.sum(mask))

    phases = [0, 100, 200, 300, 400, 500, len(meanTrace)]

    meanTracePhases = np.array([
        np.nanmean(meanTrace[phases[j]:phases[j+1]])
        for j in range(len(phases)-1)
    ])

    semTracePhases = np.array([
        np.sqrt(
            np.nanmean(
                semTrace[phases[j]:phases[j+1]]**2
            )
        )
        for j in range(len(phases)-1)
    ])

    for j in allDistance[mask]:
        axes2[1, 0].plot(j, color=colors[i-1], alpha=0.1)
        axes2[0, 2].plot([0, 1, 2, 3, 4, 5], [np.mean(j[0:100]), np.mean(j[100:200]), np.mean(j[200:300]),
                                              np.mean(j[300:400]), np.mean(j[400:500]), np.mean(j[500:])],
                         linestyle='-', color=colors[i-1], alpha=0.3)

    for j, (x, y) in enumerate(zip(allXCor[mask], allYCor[mask])):
        axes5[0].plot(x[0:100], y[0:100], color=colors[i-1], alpha=0.4)
        axes5[1].plot(x[100:200], y[100:200], color=colors[i-1], alpha=0.4)
        axes5[2].plot(x[200:300], y[200:300], color=colors[i-1], alpha=0.4)
        axes5[3].plot(x[300:400], y[300:400], color=colors[i-1], alpha=0.4)
        axes5[4].plot(x[400:500], y[400:500], color=colors[i-1], alpha=0.4)
        axes5[5].plot(x[500:], y[500:], color=colors[i-1], alpha=0.4)

    axes2[1, 2].plot(
        np.arange(len(meanTracePhases)),
        meanTracePhases,
        color=colors[i-1],
        alpha=0.7
    )

    axes2[1, 2].fill_between(
        np.arange(len(meanTracePhases)),
        meanTracePhases - semTracePhases,
        meanTracePhases + semTracePhases,
        color=colors[i-1],
        alpha=0.2
    )

    axes2[2, 0].plot(
        np.arange(len(meanTrace)),
        meanTrace,
        color=colors[i-1],
        alpha=0.7
    )

    axes2[2, 0].fill_between(
        np.arange(len(meanTrace)),
        meanTrace - semTrace,
        meanTrace + semTrace,
        color=colors[i-1],
        alpha=0.2
    )

    # outdf["Mean distance " + str(i)] = pd.Series(meanTracePhases)
    # outdf["SEM distance " + str(i)] = pd.Series(semTracePhases)

axes2[0, 0].set_ylabel("Distance (cm)")
axes2[0, 0].axvspan(100, 200, color="orange", alpha=0.2, zorder=0)
axes2[0, 0].xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: int(x/10)))
axes2[1, 0].set_ylabel("Distance (cm)")
axes2[1, 0].axvspan(100, 200, color="orange", alpha=0.2, zorder=0)
axes2[1, 0].set_ylim(0, 15)
axes2[1, 0].xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: int(x/10)))
axes2[2, 0].set_ylabel("Distance (cm)")
axes2[2, 0].axvspan(100, 200, color="orange", alpha=0.2, zorder=0)
axes2[2, 0].set_ylim(3, 15)
axes2[2, 0].xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: int(x/10)))
axes2[2, 0].set_xlabel("Time (s)")
axes2[0, 1].set_ylabel("Stim distance (cms)")
axes2[0, 1].set_ylim(0, 15)
axes2[0, 1].set_xlabel("Pre distance (cm)")
axes2[0, 1].set_xlim(0, 15)
axes2[0, 1].plot([0, 15], [0, 15], "--", color="black")
axes2[1, 1].set_ylabel("Distance change (cm)")
axes2[1, 1].set_ylim(-8, 8)
axes2[1, 1].set_xlabel("Pre distance (cms)")
axes2[1, 1].set_xlim(0, 15)
axes2[1, 1].axhline(0, ls=":")
axes2[0, 2].set_ylabel("Distance (cm)")
axes2[0, 2].set_ylim(3, 15)
axes2[0, 2].axvspan(0.5, 1.5, color="orange", alpha=0.2, zorder=0)
axes2[1, 2].set_ylabel("Distance (cm)")
axes2[1, 2].set_ylim(3, 15)
axes2[1, 2].axvspan(0.5, 1.5, color="orange", alpha=0.2, zorder=0)

if graphFormat == "svg":
    axes2[1, 0].remove()
fig2.savefig(os.path.join(
    pathfiles, f"Graph2.{graphFormat}"), format=graphFormat)

cmap = cm.get_cmap('viridis')
sortIndices = np.argsort(allpreSpeed)
sortallSpeed = np.array(allSpeed)[sortIndices]
cax1 = axes3.imshow(
    sortallSpeed,
    cmap=cmap,
    aspect="auto",
    vmin=0,
    vmax=20
)

fig3.colorbar(cax1, ax=axes3, label="Speed (cm/s)")
axes3.axvline(x=100, color="orange", linestyle="--", linewidth=1)
axes3.axvline(x=200, color="orange", linestyle="--", linewidth=1)
fig3.savefig(os.path.join(
    pathfiles, f"Graph3.{graphFormat}"), format=graphFormat)

sortIndices = np.argsort(allpreDistance)
sortallDistance = np.array(allDistance)[sortIndices]
cax3 = axes4.imshow(
    sortallDistance,
    cmap=cmap,
    aspect="auto",
    vmin=0,
    vmax=15
)

fig4.colorbar(cax3, ax=axes4, label="Distance (cm)")
axes4.axvline(x=100, color="orange", linestyle="--", linewidth=1)
axes4.axvline(x=200, color="orange", linestyle="--", linewidth=1)
fig4.savefig(os.path.join(
    pathfiles, f"Graph4.{graphFormat}"), format=graphFormat)

fig5.savefig(os.path.join(
    pathfiles, f"Graph5.{graphFormat}"), format=graphFormat)

outdf["Pre Speed"] = pd.Series(allpreSpeed)
outdf["Speed change"] = pd.Series(allSpeedChange)
outdf["Pre Distance"] = pd.Series(allpreDistance)
outdf["Distance change"] = pd.Series(allDistanceChange)

outputfileName = "Population.csv"
filePath = os.path.join(pathfiles, outputfileName)
outdf.to_csv(filePath)
print("All files have been processed")
