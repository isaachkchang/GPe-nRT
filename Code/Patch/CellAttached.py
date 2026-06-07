"""
Analzye response to a brief stimulus in cell-attached configuration.
Author: Isaac Chang
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import pyabf
import scipy
from scipy.signal import find_peaks


def load(path_in):
    files = []
    for i in os.listdir(path_in):
        if os.path.splitext(i)[-1].lower() == ".abf":
            files.append(i)
    files.sort()
    return files


def insFreq(abf, fileNum):
    abf.setSweep(0)
    apMatrix = np.empty((abf.sweepCount, 0)).tolist()
    infreqMatrix = np.zeros(
        (abf.sweepCount, int(len(abf.sweepY) / binSize))).tolist()
    time = np.arange(0, len(abf.sweepY), binSize)
    latency = -1

    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber, baseline=[0, 0.1])
        trace = abf.sweepY
        peaks, peaks_dict = find_peaks(-trace, height=-
                                       np.min(trace)*0.7, prominence=0.2 * np.max(-trace),)
        apMatrix[sweepNumber] = np.append(
            apMatrix[sweepNumber], peaks)
        axes[0].plot(
            trace + yOffset * sweepNumber,
            color=cmap(colorFactor[sweepNumber]),
            alpha=0.5,
        )
        for index, value in enumerate(apMatrix[sweepNumber]):
            axes[0].plot(
                int(value),
                trace[int(value)] + yOffset * sweepNumber,
                color="red",
                marker="o",
                markersize=1,
            )

        if sweepNumber == 0 and len(np.where(apMatrix[sweepNumber] > stimStart)[0]) != 0:
            psth[fileNum] = apMatrix[sweepNumber]
            firstAPPos = np.where(apMatrix[sweepNumber] > stimStart)[0][0]
            axes[0].axvline(apMatrix[sweepNumber][firstAPPos],
                            ls=":", color="orange")
            latency = (apMatrix[sweepNumber][firstAPPos] -
                       stimStart)/abf.dataPointsPerMs
            if latency < 4:
                latency = (apMatrix[sweepNumber][firstAPPos+1] -
                           stimStart)/abf.dataPointsPerMs
            psthLatency[fileNum] = latency

        timeDiff = np.diff(apMatrix[sweepNumber])
        for index, value in enumerate(apMatrix[sweepNumber][:-1]):
            for index2, value2 in enumerate(
                time[int(value / binSize)
                         : int((value + timeDiff[index]) / binSize)]
            ):
                infreqMatrix[sweepNumber][index2 + int(value / binSize)] = 1 / (
                    timeDiff[index] / abf.dataRate
                )

    for index, value in enumerate(infreqMatrix):
        axes[1].plot(np.array(value), color=cmap(
            colorFactor[index]), alpha=0.7)

    infreqmeanTrace = np.mean(np.array(infreqMatrix), axis=0)
    frbsl = np.mean(
        infreqmeanTrace[
            int((stimStart - preStimWindow) / binSize): int(stimStart / binSize)
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
        axes[2].axhline(y=frMax, color="red", linestyle="--")
        frChange = frMax - frbsl
    else:
        axes[2].axhline(y=frMin, color="red", linestyle="--")
        frChange = frMin - frbsl
    axes[2].plot(time, infreqmeanTrace)
    axes[2].axhline(y=frbsl, color="blue", linestyle="--")

    return frChange, latency


path_in = r"C:\Users\isaac.chang\Desktop\Data\Patch\KCl_Opto\CellAttached"
path_out = path_in
pathinfo = path_in

cellInfo = pd.read_csv(os.path.join(pathinfo, "info.csv"))
firing = cellInfo["Effect"].tolist()

files = load(path_in)
psth = np.empty((len(files), 0)).tolist()
psthLatency = np.empty(len(files))

df = pd.DataFrame()

print("Analyzing " + str(len(files)) + " files below:")
print(files)

for index, value in enumerate(files):
    abf = pyabf.ABF(os.path.join(path_in, value))

    stimStart = 2078.12 * abf.dataPointsPerMs
    preStimWindow = 500 * abf.dataPointsPerMs
    postStimWindow = 100 * abf.dataPointsPerMs
    binSize = 10 * abf.dataPointsPerMs
    yOffset = -100
    cmap = plt.cm.get_cmap("viridis")
    colorFactor = np.linspace(0, 1, abf.sweepCount)

    fig, axes = plt.subplots(3)
    fig.set_figheight(14)
    fig.set_figwidth(7)

    frChange, latency = insFreq(abf, index)

    df = df.append(pd.Series([], dtype="float64", name=index))
    df.loc[index, "File name"] = str(os.path.basename(abf.abfFilePath))
    df.loc[index, "Stim firing rate change"] = frChange
    df.loc[index, "Latency to burst"] = latency

    axes[0].axis("off")
    axes[1].set_ylabel("Instantaneous freq (Hz)")
    axes[1].axvspan(stimStart/binSize, (stimStart + 2 * abf.dataPointsPerMs)/binSize,
                    alpha=0.3, color="skyblue")
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x)/100)))
    axes[1].set_xlabel("Time (s)")
    axes[2].set_ylabel("Instantaneous freq (Hz)")
    axes[2].axvspan(stimStart, stimStart + 2 * abf.dataPointsPerMs,
                    alpha=0.3, color="skyblue")
    axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x)/abf.dataRate)))
    axes[2].set_xlabel("Time (s)")

    plt.savefig(os.path.join(path_out, value + " _Graph.png"), format="png")
    plt.close()
    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")


indicepsthLatency = np.argsort(-psthLatency)
psth = [psth[i] for i in indicepsthLatency]
firing = [firing[i] for i in indicepsthLatency]

fig, axes = plt.subplots(2)
fig.set_figheight(7)
fig.set_figwidth(7)

for index, value in enumerate(psth):
    if firing[index] == "active increase" or firing[index] == "silent increase":
        axes[0].scatter(value, [index]*len(value), color="darkblue", s=1)
        axes[1].scatter(value, [index]*len(value), color="darkblue", s=1)
    elif firing[index] == "no change":
        axes[0].scatter(value, [index]*len(value), color="grey", s=1)
        axes[1].scatter(value, [index]*len(value), color="grey", s=1)
    elif firing[index] == "decrease":
        axes[0].scatter(value, [index]*len(value), color="darkred", s=1)
        axes[1].scatter(value, [index]*len(value), color="darkred", s=1)

axes[0].axvspan(207812, 208012, alpha=0.3, color="skyblue")
axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: float((x)/abf.dataRate)))
axes[1].axvspan(207812, 208012, alpha=0.3, color="skyblue")
axes[1].set_xlim(200812, 215812)
axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: float((x)/abf.dataRate)))

plt.savefig(os.path.join(path_out, "PSTH.png"), format="svg")
plt.close()

fileName = "CellAttached_Results.csv"
filePath = os.path.join(path_out, fileName)
df.to_csv(filePath)
print("All files have been processed")
