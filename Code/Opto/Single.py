"""
Process DLC filtered data files for opto individual mouse data.
Author: Isaac Chang
"""

import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
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
    
    bins = np.logspace(-0.3, 0.5, 5)
    bins = np.concatenate(([0], bins, [np.inf]))
    colors = cmap(np.linspace(0.5, 1, len(bins)-1))[::-1]

    for i in np.arange(optoTrial):
        start = int(firstoptoFrames[index] + i * lengthTrial)
        end = int(firstoptoFrames[index] + (i + 1) * lengthTrial)
        optoData[i] = data[start:end]
        preSpeed[i] = np.mean(optoData[i][0:100])
        stimSpeed[i] = np.mean(optoData[i][100:200])
        speedChange[i] = stimSpeed[i] / preSpeed[i]
        axes[0, 2].plot(
            preSpeed[i], speedChange[i], "o", markersize=2, color="blue"
        )

        binID = np.digitize(preSpeed[i], bins) - 1
        color = colors[binID]
        axes[2, 0].plot(optoData[i], color=color, alpha = 0.2)

    sortIndices = np.argsort(preSpeed)
    sortoptoData = optoData[sortIndices]
    cax1 = axes[0, 0].imshow(
        sortoptoData,
        cmap=cmap2,
        aspect="auto",
        interpolation='none'
    )
    
    plt.colorbar(cax1, ax=axes[0, 0], label="Speed (cm/s)")

    avgSpeedX = np.mean(optoData, axis=0)
    semSpeedX = np.std(optoData, axis=0, ddof=1) / np.sqrt(optoTrial)
    axes[1, 0].plot(avgSpeedX)
    axes[1, 0].fill_between(
        np.arange(0, lengthTrial),
        avgSpeedX - semSpeedX,
        avgSpeedX + semSpeedX,
        alpha=0.3)

    return 

def distanceAnalyze(dataX, dataY):

    optoData = np.empty((optoTrial, lengthTrial))
    preDistance = np.empty(optoTrial)
    stimDistance = np.empty(optoTrial)
    distanceChange = np.empty(optoTrial)

    XCenter = (np.max(dataX)+np.min(dataX))/2
    YCenter = (np.max(dataY)+np.min(dataY))/2

    bins = np.linspace(7, 14, 8)
    bins = np.concatenate(([0], bins, [np.inf]))
    colors = cmap(np.linspace(0.5, 1, len(bins)-1))[::-1]

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

        optoData[i] = distance
        preDistance[i] = np.mean(distance[0:100])
        stimDistance[i] = np.mean(distance[100:200])
        distanceChange[i] = stimDistance[i] - preDistance[i]

        axes[1, 2].plot(
            preDistance[i], distanceChange[i], "o", markersize=2, color="blue"
        )

        binID = np.digitize(preDistance[i], bins) - 1
        color = colors[binID]
        axes[2, 1].plot(optoData[i], color=color, alpha = 0.5)

    sortIndices = np.argsort(preDistance)
    sortoptoData = optoData[sortIndices]
    cax2 = axes[0, 1].imshow(
        sortoptoData,
        cmap=cmap2,
        aspect="auto",
        interpolation='none',
        vmin=0,
        vmax=15
    )

    plt.colorbar(cax2, ax=axes[0, 1], label="Distance")

    avgDistance = np.mean(optoData, axis=0)
    semDistance = np.std(optoData, axis=0, ddof=1) / np.sqrt(optoTrial)
    axes[1, 1].plot(avgDistance)
    axes[1, 1].fill_between(
        np.arange(0, lengthTrial),
        avgDistance - semDistance,
        avgDistance + semDistance,
        alpha=0.3,
    )

    return 

optoTrial = 20

pathfiles = r"\\hive\Paz-Lab\Isaac\GPe\Data\Opto\Analysis\SOM-eYFP-20Hz"
pathinfo = pathfiles
videoInfo = pd.read_csv(os.path.join(pathinfo, "videoInfo.csv"))
files = load()
files = [f for f in files if f[:-4] in set(videoInfo["Video file"])]
firstoptoFrames = videoInfo["First opto frame"].tolist()
lastoptoFrames = videoInfo["Last opto frame"].tolist()

print("Analyzing " + str(len(files)) + " files below:")
print(files)

for index, value in enumerate(files):
    df = pd.read_csv(os.path.join(pathfiles, value))
    speed = df["speed"].to_numpy()
    XCor = df["XCor"].to_numpy()
    YCor = df["YCor"].to_numpy()

    lengthTrial = int(
        (lastoptoFrames[index] - firstoptoFrames[index]) / (optoTrial-1))

    fig, axes = plt.subplots(
        3, 3, figsize=(20, 10), gridspec_kw={"width_ratios": [3, 3, 1]}
    )

    cmap = cm.get_cmap('Blues')
    cmap2 = cm.get_cmap("viridis")

    speedAnalyze(speed)
    distanceAnalyze(XCor, YCor)

    axes[0, 0].set_ylabel("Optogenetic Trial")
    axes[0, 0].axvline(x=100, color="orange", linestyle="--", linewidth=1)
    axes[0, 0].axvline(x=200, color="orange", linestyle="--", linewidth=1)
    axes[0, 0].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int(x/10)))
    axes[1, 0].set_ylabel("Speed (cm/s)")
    axes[1, 0].axvspan(100, 200, color="orange", alpha=0.2, zorder=0)
    axes[1, 0].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int(x/10)))
    axes[2, 0].set_ylabel("Speed (cm/s)")
    axes[2, 0].axvspan(100, 200, color="orange", alpha=0.2, zorder=0)
    axes[2, 0].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int(x/10)))
    axes[2, 0].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Optogenetic Trial")
    axes[0, 1].axvline(x=100, color="orange", linestyle="--", linewidth=1)
    axes[0, 1].axvline(x=200, color="orange", linestyle="--", linewidth=1)
    axes[0, 1].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int(x/10)))
    axes[1, 1].set_ylabel("Distance (cm)")
    axes[1, 1].axvspan(100, 200, color="orange", alpha=0.2, zorder=0)
    axes[1, 1].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int(x/10)))
    axes[2, 1].set_ylabel("Distance (cm)")
    axes[2, 1].axvspan(100, 200, color="orange", alpha=0.2, zorder=0)
    axes[2, 1].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int(x/10)))
    axes[2, 1].set_xlabel("Time (s)")
    axes[0, 2].set_ylabel("Fold change")
    axes[0, 2].set_ylim(0.01, 100)
    axes[0, 2].set_yscale('log')
    axes[0, 2].set_xlabel("Pre speed (cm/s)")
    axes[0, 2].set_xlim(0, 10)
    axes[0, 2].axhline(1, ls=":")
    axes[1, 2].set_ylabel("Change (cm)")
    axes[1, 2].set_ylim(-8, 8)
    axes[1, 2].set_xlabel("Pre distance (cms)")
    axes[1, 2].set_xlim(0, 15)
    axes[1, 2].axhline(0, ls=":")

    plt.savefig(os.path.join(pathfiles, value + " _Graph.png"), format="png")
    plt.close()
    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")

print("All files have been processed")
