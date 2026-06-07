"""
Analyze response to prolonged opsin activation during rest
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


def calculateVm(abf):

    assert isinstance(abf, pyabf.ABF)

    abf.setSweep(0)
    trace = abf.sweepY
    axes.plot(trace)
    apPeakPos = findAPPos(trace[stepStart:stepStart + stepDur])

    numAP = len(apPeakPos)
    preVm = np.mean(trace[stepStart-preStimWindow:stepStart])
    maxVm = np.max(trace[stepStart:stepStart + stepDur])
    minVm = np.min(trace[stepStart:stepStart + stepDur])
    if abs(maxVm - preVm) >= abs(minVm - preVm):
        opto1Vm = maxVm
    else:
        opto1Vm = minVm

    opto2Vm = np.mean(
        trace[stepStart + stepDur - 100*abf.dataPointsPerMs: stepStart + stepDur])
    postVm = np.mean(
        trace[stepStart + stepDur: stepStart + stepDur + 100 * abf.dataPointsPerMs])

    axes.axvline(np.where(trace[stepStart:stepStart + stepDur]
                 == opto1Vm)[0][0] + stepStart, color="orange")

    return numAP, preVm, opto1Vm, opto2Vm, postVm


path_in = r"C:\Users\isaac.chang\Desktop\Data\Patch\Kgluc_Opto\nRT opsin\OpsinSilent"
path_out = path_in

files = load(path_in)


df = pd.DataFrame()

print("Analyzing " + str(len(files)) + " files below:")
print(files)

for index, value in enumerate(files):
    abf = pyabf.ABF(os.path.join(path_in, value))

    VThreshold = -10
    stepStart = int(5312.5*abf.dataPointsPerMs)
    stepDur = int(10000 * abf.dataPointsPerMs)
    preStimWindow = int(10 * abf.dataPointsPerMs)

    fig, axes = plt.subplots(1, 1)
    fig.set_figheight(6)
    fig.set_figwidth(10)

    numAP, preVm, opto1Vm, opto2Vm, postVm = calculateVm(abf)

    df = df.append(pd.Series([], dtype="float64", name=index))
    df.loc[index, "File name"] = str(os.path.basename(abf.abfFilePath))
    df.loc[index, "Max PSP"] = opto1Vm - preVm
    df.loc[index, "Steady-state PSP"] = opto2Vm - preVm

    axes.set_ylabel("Vm (mV)")
    axes.axvspan(stepStart, stepStart + stepDur, alpha=0.3, color="skyblue")
    axes.set_xlabel("Time (s)")
    axes.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x)/abf.dataRate)))
    plt.savefig(os.path.join(path_out, value + "_Graph.png"))
    plt.close()

    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")

fileName = "OpsinSilent_Results.csv"
filePath = os.path.join(path_out, fileName)
df.to_csv(filePath)
print("All files have been processed")
