"""
Analyze response to prolonged opsin activation during firing
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

    apPeakBin = np.where(trace >= VThreshold)[0]
    apPeakBin = np.concatenate(([0], apPeakBin))
    apPeakBin2 = apPeakBin[:-1] - apPeakBin[1:] + 1
    apPeakBin3 = apPeakBin[np.where(apPeakBin2)[0] + 1]

    return apPeakBin3


def calculateAP(abf):
    assert isinstance(abf, pyabf.ABF)

    abf.setSweep(0)
    trace = abf.sweepY[preStart:postStart + stepDur]
    apPeakPos = findAPPos(trace)

    axes[0].plot(abf.sweepY, linewidth=0.5)
    for index, value in enumerate(apPeakPos):
        axes[0].plot(int(preStart+value),
                     abf.sweepY[int(preStart + value)], color="green", marker="o")

    preAP = apPeakPos[(apPeakPos < stepDur)].size
    optoAP = apPeakPos[(apPeakPos >= stepDur) &
                       (apPeakPos < 2*stepDur)].size
    postAP = apPeakPos[(apPeakPos >= 2*stepDur) &
                       (apPeakPos < 3*stepDur)].size

    Vm = []

    for i in range(stepSize - 1, len(trace), stepSize):
        avgVm = np.mean(trace[i-preStimWindow:i])
        Vm.append(avgVm)

    preVm = np.mean(Vm[95:99])
    maxVm = np.max(Vm[100:199])
    minVm = np.min(Vm[100:199])
    if abs(maxVm - preVm) >= abs(minVm - preVm):
        opto1Vm = maxVm
    else:
        opto1Vm = minVm
    opto2Vm = np.mean(Vm[195:199])
    postVm = np.mean(Vm[199:203])

    axes[1].plot(Vm)
    axes[1].axvline(np.where(Vm[100:199] == opto1Vm)
                    [0][0] + 100, color="orange")

    return preAP, optoAP, postAP, preVm, opto1Vm, opto2Vm, postVm


path_in = r"C:\Users\isaac.chang\Desktop\Data\Patch\Kgluc_Opto\nRT opsin\OpsinFiring"
path_out = path_in

files = load(path_in)

df = pd.DataFrame()

print("Analyzing " + str(len(files)) + " files below:")
print(files)

for index, value in enumerate(files):
    abf = pyabf.ABF(os.path.join(path_in, value))

    VThreshold = -10
    stepSize = int(200 * abf.dataPointsPerMs)
    stepDur = int(20000 * abf.dataPointsPerMs)
    preStart = int(6093.7 * abf.dataPointsPerMs)
    optoStart = int(26093.7 * abf.dataPointsPerMs)
    postStart = int(46093.7 * abf.dataPointsPerMs)
    preStimWindow = int(10 * abf.dataPointsPerMs)

    fig, axes = plt.subplots(2, 1)
    fig.set_figheight(12)
    fig.set_figwidth(12)

    preAP, optoAP, postAP, preVm, opto1Vm, opto2Vm, postVm = calculateAP(abf)

    df = df.append(pd.Series([], dtype="float64", name=index))
    df.loc[index, "File name"] = str(os.path.basename(abf.abfFilePath))
    df.loc[index, "preAP"] = preAP
    df.loc[index, "optoAP"] = optoAP
    df.loc[index, "postAP"] = postAP
    df.loc[index, "Max PSP"] = opto1Vm - preVm
    df.loc[index, "Steady-state PSP"] = opto2Vm - preVm
    df.loc[index, "Rebound PSP"] = postVm - preVm

    axes[0].set_ylabel("Vm (mV)")
    axes[0].axvspan(optoStart, preStart, alpha=0.3, color="skyblue")
    axes[0].set_xlabel("Time (s)")
    axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x)/abf.dataRate)))
    axes[1].set_ylabel("Vm (mV)")
    axes[1].axvspan(99, 199, alpha=0.3, color="skyblue")
    axes[1].set_xlabel("Time (s)")
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int(x/5)))

    plt.savefig(os.path.join(path_out, value + "_Graph.png"))

    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")

fileName = "OpsinFiring_Results.csv"
filePath = os.path.join(path_out, fileName)
df.to_csv(filePath)
print("All files have been processed")
