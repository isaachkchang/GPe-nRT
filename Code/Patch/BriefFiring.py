"""
Analzye response to a brief stimulus during firing.
Author: Isaac Chang
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import pyabf


def load(path_in):
    files = []
    for i in os.listdir(path_in):
        if os.path.splitext(i)[-1].lower() == ".abf":
            files.append(i)
    files.sort()
    return files


def findAPPos(trace):
    dtrace = np.diff(trace)
    apPeakBin = np.where(dtrace >= dVThreshold)[0]
    apPeakBin = np.concatenate(([0], apPeakBin))
    apPeakBin2 = apPeakBin[:-1] - apPeakBin[1:] + 1
    apPeakBin3 = apPeakBin[np.where(apPeakBin2)[0] + 1]
    apPeakPos = []

    if len(apPeakBin3):
        apPeakPos.append(apPeakBin3[0])
        for i in range(1, len(apPeakBin3)):
            if apPeakBin3[i] - apPeakBin3[i - 1] >= dTThreshold:
                apPeakPos.append(apPeakBin3[i])

    return apPeakPos


def insFreq(abf, stepNum):

    stepStart = step[stepNum][0]
    stepEnd = step[stepNum][1]
    apMatrix = np.empty((abf.sweepCount, 0)).tolist()
    infreqMatrix = np.zeros(
        (abf.sweepCount, int((stepEnd - stepStart) / binSize))
    ).tolist()
    time = np.arange(0, stepEnd - stepStart, binSize)

    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber)
        trace = abf.sweepY[stepStart:stepEnd]
        apMatrix[sweepNumber] = np.append(
            apMatrix[sweepNumber], [findAPPos(trace)])
        axes[0][stepNum].plot(
            trace + yOffset * sweepNumber,
            color=cmap(colorFactor[sweepNumber]),
            alpha=0.5,
        )
        for index, value in enumerate(apMatrix[sweepNumber]):
            axes[0][stepNum].plot(
                int(value),
                trace[int(value)] + yOffset * sweepNumber,
                color="red",
                marker="o",
                markersize=1,
            )

        timeDiff = np.diff(apMatrix[sweepNumber])
        for index, value in enumerate(apMatrix[sweepNumber][:-1]):
            for index2, value2 in enumerate(
                time[int(value / binSize)                     : int((value + timeDiff[index]) / binSize)]
            ):
                infreqMatrix[sweepNumber][index2 + int(value / binSize)] = 1 / (
                    timeDiff[index] / abf.dataRate
                )

    for index, value in enumerate(infreqMatrix):
        axes[1][stepNum].plot(
            np.array(value), color=cmap(colorFactor[index]), alpha=0.7
        )

    infreqmeanTrace = np.mean(np.array(infreqMatrix), axis=0)
    frbsl = np.mean(
        infreqmeanTrace[
            int((stimStart - 2 * preStimWindow) / binSize): int(
                (stimStart - preStimWindow) / binSize
            )
        ]
    )
    frMax = np.max(
        infreqmeanTrace[
            int(stimStart / binSize): int((stimStart + postStimWindow) / binSize)
        ]
    )
    frMin = np.min(
        (
            infreqmeanTrace[
                int(stimStart / binSize): int((stimStart + postStimWindow) / binSize)
            ]
        )
    )
    if frMax - frbsl > frbsl - frMin:
        axes[2][stepNum].axhline(y=frMax, color="red", linestyle="--")
        frChange = frMax - frbsl
    else:
        axes[2][stepNum].axhline(y=frMin, color="red", linestyle="--")
        frChange = frMin - frbsl
    relfrChange = frChange / frbsl * 100
    axes[2][stepNum].plot(time, infreqmeanTrace)
    axes[2][stepNum].axhline(y=frbsl, color="blue", linestyle="--")

    return frChange, relfrChange


path_in = r"\\hive\Paz-Lab\Isaac\GPe\Data\Patch\Kgluc_Opto\nRT\BriefFiring"
path_out = path_in

files = load(path_in)

df = pd.DataFrame()

print("Analyzing " + str(len(files)) + " files below:")
print(files)

for index, value in enumerate(files):
    abf = pyabf.ABF(os.path.join(path_in, value))

    stimStart = 1000 * abf.dataPointsPerMs
    dTThreshold = 2 * abf.dataPointsPerMs
    dVThreshold = 0.5
    step = [
        [int(356.25 * abf.dataPointsPerMs), int(2356.25 * abf.dataPointsPerMs)],
        [int(7356.25 * abf.dataPointsPerMs),
         int(9356.25 * abf.dataPointsPerMs)],
    ]
    preStimWindow = 50 * abf.dataPointsPerMs
    postStimWindow = 100 * abf.dataPointsPerMs
    binSize = 1 * abf.dataPointsPerMs
    cmap = plt.cm.get_cmap("viridis")
    colorFactor = np.linspace(0, 1, abf.sweepCount)
    yOffset = -100

    fig, axes = plt.subplots(3, 2)
    fig.set_figheight(10)
    fig.set_figwidth(10)

    frChange1, relfrChange1 = insFreq(abf, 0)
    frChange2, relfrChange2 = insFreq(abf, 1)

    df = df.append(pd.Series([], dtype="float64", name=index))
    df.loc[index, "File name"] = str(os.path.basename(abf.abfFilePath))
    df.loc[index, "Stim firing rate change"] = frChange1
    df.loc[index, "NoStim firing rate change"] = frChange2
    df.loc[index, "Stim relative firing rate change"] = relfrChange1
    df.loc[index, "NoStim relative firing rate change"] = relfrChange2

    axes[0, 0].axis("off")
    axes[0, 1].axis("off")
    axes[1, 0].set_ylabel("Instantaneous freq (Hz)")
    axes[1, 0].axvspan(stimStart/binSize, (stimStart + 2 * abf.dataPointsPerMs)/binSize,
                       alpha=0.3, color="skyblue")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 1].set_ylabel("Instantaneous freq (Hz)")
    axes[1, 1].axvspan(stimStart/binSize, (stimStart + 2 * abf.dataPointsPerMs)/binSize,
                       alpha=0.3, color="skyblue")
    axes[1, 1].set_xlabel("Time (ms)")
    axes[2, 0].set_ylabel("Instantaneous freq (Hz)")
    axes[2, 0].axvspan(stimStart, stimStart + 2 * abf.dataPointsPerMs,
                       alpha=0.3, color="skyblue")
    axes[2, 0].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x)/abf.dataPointsPerMs)))
    axes[2, 0].set_xlabel("Time (ms)")
    axes[2, 1].set_ylabel("Instantaneous freq (Hz)")
    axes[2, 1].axvspan(stimStart, stimStart + 2 * abf.dataPointsPerMs,
                       alpha=0.3, color="skyblue")
    axes[2, 1].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x)/abf.dataPointsPerMs)))
    axes[2, 1].set_xlabel("Time (ms)")

    plt.savefig(os.path.join(path_out, value + " _Graph.png"), format="png")
    plt.close()

    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")

fileName = "BriefFiring_Results.csv"
filePath = os.path.join(path_out, fileName)
df.to_csv(filePath)
print("All files have been processed")
