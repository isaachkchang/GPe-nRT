"""
Analzye response to a brief stimulus in cell-attached configuration, with drug wash-on.
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

    apMatrix = np.empty((abf.sweepCount, 0)).tolist()
    latency = np.full(abf.sweepCount, np.nan)
    yOffset = -100

    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber, baseline=[0, 0.1])
        trace = abf.sweepY
        peaks, peaks_dict = find_peaks(-trace, height=-
                                       np.min(trace)*0.7, prominence=0.2 * np.max(-trace),)
        apMatrix[sweepNumber] = np.append(
            apMatrix[sweepNumber], peaks)

        for index, value in enumerate(apMatrix[sweepNumber]):
            axes.plot(
                int(value),
                trace[int(value)] + yOffset * sweepNumber,
                color="black",
                marker="o",
                markersize=1,
            )

        firstAPPos = np.where(apMatrix[sweepNumber] > stimStart)[0]
        if len(firstAPPos) > 0:
            latency[sweepNumber] = (
                apMatrix[sweepNumber][firstAPPos[0]] - stimStart
            ) / abf.dataPointsPerMs

    return latency


path_in = r"C:\Users\isaac.chang\Desktop\Data\Patch\Kgluc_Opto\nRT VU\DrugWashOn"
path_out = path_in

files = load(path_in)

df = pd.DataFrame()

print("Analyzing " + str(len(files)) + " files below:")
print(files)

for index, value in enumerate(files):
    abf = pyabf.ABF(os.path.join(path_in, value))

    cmap = plt.cm.get_cmap("viridis")
    stimStart = 2078.12 * abf.dataPointsPerMs
    yOffset = -100

    fig, axes = plt.subplots(1)
    fig.set_figheight(5)
    fig.set_figwidth(5)

    latency = insFreq(abf, index)

    df = df.append(pd.Series([], dtype="float64", name=index))
    df.loc[index, "File name"] = str(os.path.basename(abf.abfFilePath))
    df.loc[index, "pre latency"] = np.nanmean(latency[:16])
    df.loc[index, "post latency"] = np.nanmean(latency[-16:])
    for index2, value2 in enumerate(latency):
        df.loc[index, str(index2)] = value2

    axes.axvspan(stimStart, stimStart + 2 * abf.dataPointsPerMs,
                 alpha=0.3, color="skyblue")
    axes.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: float((x)/abf.dataRate)))
    axes.set_xlabel("Time (s)")

    plt.savefig(os.path.join(path_out, value + " _Graph.svg"), format="svg")
    plt.close()
    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")

fileName = "DrugWashOn_Results.csv"
filePath = os.path.join(path_out, fileName)
df.to_csv(filePath)
print("All files have been processed")
