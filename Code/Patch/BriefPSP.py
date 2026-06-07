"""
Analzye response to a brief stimulus during rest.
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


def IVCurve(abf):

    Vm = np.zeros(len(abf.sweepList))
    apMatrix = np.empty((abf.sweepCount, 0)).tolist()
    apNum = np.zeros(len(abf.sweepList))
    EPSPamp = np.full(len(abf.sweepList), np.nan)
    latency = np.full(len(abf.sweepList), np.nan)
    binEPSPamp = np.full(len(bin), np.nan)
    binlatency = np.full(len(bin), np.nan)
    binapNum = np.zeros(len(bin))
    peakPos = np.nan
    EClx = np.nan

    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber)
        sweep = abf.sweepY - junctionPotential
        Vm[sweepNumber] = np.mean(sweep[stimOnset - preStimWindow: stimOnset])
        trace = sweep[stimOnset: stimOnset + postStimWindow]
        apMatrix[sweepNumber] = np.append(
            apMatrix[sweepNumber], [findAPPos(trace)])
        axes[0].plot(
            sweep[stimOnset - preStimWindow: stimOnset + postStimWindow]
            + yOffset * sweepNumber,
            color=cmap(colorFactor[sweepNumber]),
            alpha=0.5,
        )
        axes[0].annotate(
            f"{Vm[sweepNumber]:.2f}",
            (0, Vm[sweepNumber] + yOffset * sweepNumber),
            textcoords="offset points",
            xytext=(0, 10),
            ha="right",
        )
        for index, value in enumerate(apMatrix[sweepNumber]):
            axes[0].plot(
                int(value) + preStimWindow,
                trace[int(value)] + yOffset * sweepNumber,
                color="red",
                marker="o",
                markersize=2,
            )

        if len(apMatrix[sweepNumber]) == 0:
            if np.isnan(peakPos):
                peak = np.max(trace)
                peakPos = np.where(trace == peak)[0][0]
            EPSPamp[sweepNumber] = trace[peakPos] - Vm[sweepNumber]
            axes[0].plot(
                peakPos + preStimWindow,
                trace[peakPos] + yOffset * sweepNumber,
                color="green",
                marker="o",
                markersize=2,
            )
        else:
            apNum[sweepNumber] = len(apMatrix[sweepNumber])
            latency[sweepNumber] = apMatrix[sweepNumber][0]
            axes[0].plot(
                latency[sweepNumber] + preStimWindow,
                trace[int(latency[sweepNumber])] + yOffset * sweepNumber,
                color="green",
                marker="o",
                markersize=2,
            )

    sortIndices = np.argsort(Vm)
    Vm.sort()
    EPSPamp = EPSPamp[sortIndices]
    apNum = apNum[sortIndices]
    latency = latency[sortIndices]
    axes[1].plot(Vm, EPSPamp)
    if EPSPamp[-1] <= 0:
        ECl2 = np.where(EPSPamp <= 0)[0][0]
        ECly = EPSPamp[ECl2 - 1] - EPSPamp[ECl2]
        EClx = Vm[ECl2 - 1] + (EPSPamp[ECl2 - 1] / ECly) * (
            EPSPamp[ECl2 - 1] - EPSPamp[ECl2]
        )
        axes[1].axvline(x=EClx, color="Blue", linestyle="--")
        axes[1].annotate(
            "ECl = " + str(EClx),
            (EClx, 0),
            textcoords="offset points",
            xytext=(0, 10),
            ha="left",
        )

    count = 0
    for index, value in enumerate(bin):
        EPSPsum = 0
        apsum = 0
        latencysum = 0
        binCount = 0
        for j in range(count, len(EPSPamp)):
            if Vm[j] >= value - binWidth / 2 and Vm[j] <= value + binWidth / 2:
                EPSPsum += EPSPamp[j]
                apsum += apNum[j]
                latencysum += latency[j]
                binCount += 1
                count += 1
        if binCount:
            if apsum > 0:
                binapNum[index] = apsum / binCount
                binlatency[index] = latencysum / binCount
            binEPSPamp[index] = EPSPsum / binCount

    binlatency = binlatency/abf.dataPointsPerMs

    return binEPSPamp, binapNum, binlatency, EClx


path_in = r"C:\Users\isaac.chang\Desktop\Data\Patch\Kgluc_Opto\nRT\BriefPSP"
path_out = path_in

files = load(path_in)

df = pd.DataFrame()

print("Analyzing " + str(len(files)) + " files below:")
print(files)

for index, value in enumerate(files):
    abf = pyabf.ABF(os.path.join(path_in, value))

    stimOnset = int(1262.5 * abf.dataPointsPerMs)
    dTThreshold = 2 * abf.dataPointsPerMs
    dVThreshold = 0.5
    preStimWindow = int(5 * abf.dataPointsPerMs)
    postStimWindow = int(200 * abf.dataPointsPerMs)
    minVm = -120
    maxVm = -30
    binWidth = 10
    bin = np.arange(minVm, maxVm, binWidth)
    junctionPotential = 15
    cmap = plt.cm.get_cmap("viridis")
    colorFactor = np.linspace(0, 1, abf.sweepCount)
    yOffset = 30

    fig, axes = plt.subplots(1, 3)
    fig.set_figheight(5)
    fig.set_figwidth(15)

    binEPSPamp, binapNum, binlatency, EClx = IVCurve(abf)

    axes[2].plot(bin, binapNum)

    df = df.append(pd.Series([], dtype="float64", name=index))
    df.loc[index, "File name"] = str(os.path.basename(abf.abfFilePath))
    df.loc[index, "ECl"] = EClx
    for index2, value2 in enumerate(bin):
        df.loc[index, "# AP evoked" + str(value2)] = binapNum[index2]

    axes[0].axis("off")
    axes[1].set_ylabel("PSP (mV)")
    axes[1].axhline(y=0, color="gray", linestyle="--")
    axes[1].axvline(x=0, color="gray", linestyle="--")
    axes[1].set_xlabel("Vm (mV)")
    axes[1].set_xlim(-110, -10)
    axes[2].set_ylabel("# AP")
    axes[2].set_xlabel("Vm (mV)")
    axes[2].set_xlim(-110, -10)

    plt.savefig(os.path.join(path_out, value + "_Graph.png"))
    plt.close()

    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")

fileName = "BriefPSP_Results.csv"
filePath = os.path.join(path_out, fileName)
df.to_csv(filePath)
print("All files have been processed")
