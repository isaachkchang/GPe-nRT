"""
Analzye response to +5mV test pulse
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
from scipy.optimize import curve_fit


def load(path_in):
    files = []
    for i in os.listdir(path_in):
        if os.path.splitext(i)[-1].lower() == ".abf":
            files.append(i)
    files.sort()
    return files


def monoExp(t, A0, A1, tau):
    return A1 * np.exp(-t / tau) + A0


def getMeanSweep(abf):
    assert isinstance(abf, pyabf.ABF)

    meanSweep = np.zeros(len(abf.sweepY))
    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber)
        meanSweep += abf.sweepY
    meanSweep = meanSweep / abf.sweepCount

    return meanSweep


def step_calculate(abf, trace):
    assert isinstance(abf, pyabf.ABF)

    Ih = np.mean(trace[:stepStart])
    stepTrace = trace[stepStart:stepEnd] - Ih
    Istep = np.mean(stepTrace[int(len(stepTrace) * 0.5):])
    Ipeak = np.max(stepTrace)
    Ra = abs(((dV * (1e-3)) / (Ipeak * (1e-12))) * (1e-6))
    Rm = abs(dV) * (1e-3) / (Istep * (1e-12)) * (1e-6) - Ra

    stepTrace = stepTrace[: abf.dataPointsPerMs * 10]
    tracePeak = np.max(stepTrace)
    tracePeakPos = np.where(stepTrace == tracePeak)[0][0]
    stepTrace = stepTrace[tracePeakPos:]
    stepTrace -= Istep
    zeroI = np.where(stepTrace <= 0)[0]
    if len(zeroI):
        zeroI = zeroI[0]
        stepTrace = stepTrace[:zeroI]
    upperFracVal = fitToFracUpper * stepTrace[0]
    upperI = np.where(stepTrace < upperFracVal)[0][0]
    fitThis = stepTrace[upperI:]
    [A0, A1, tau], cv = scipy.optimize.curve_fit(
        monoExp, np.arange(len(fitThis)), fitThis
    )
    tau /= abf.dataPointsPerMs
    Cm = tau * (1e-3) / (Ra * (1e-6))

    ax.plot(trace)
    ax.plot(
        int(stepStart + tracePeakPos),
        trace[int(stepStart + tracePeakPos)],
        marker="o",
        markersize=5,
    )
    ax.plot(
        int(np.where(trace >= (Istep + Ih))[0][0]),
        trace[int(np.where(trace >= (Istep + Ih))[0][0])],
        marker="o",
        markersize=5,
    )
    ax.plot(
        np.arange(len(fitThis)) + stepStart + tracePeakPos,
        monoExp(np.arange(len(fitThis)), A0, A1, tau * abf.dataPointsPerMs)
        + Ih
        + Istep,
        "--",
        color="orange",
    )

    return Ih, Rm, Ra, Cm, tau


path_in = r"C:\Users\isaac.chang\Desktop\Data\Patch\Kgluc_Opto\nRT opsin\MemTest"
path_out = path_in

files = load(path_in)
print("Analyzing " + str(len(files)) + " files below:")
print(files)

df = pd.DataFrame()

for index, value in enumerate(files):
    abf = pyabf.ABF(os.path.join(path_in, value))

    dV = 5
    fitToFracUpper = 0.9
    stepStart = int(106.3 * abf.dataPointsPerMs)
    stepEnd = int(206.3 * abf.dataPointsPerMs)

    fig, (ax) = plt.subplots(1, 1)
    fig.set_figheight(4)
    fig.set_figwidth(6)

    meanSweep = getMeanSweep(abf)
    Ih, Rm, Ra, Cm, tau = step_calculate(abf, meanSweep)

    df = pd.concat([df, pd.Series(dtype="float64", name=index).to_frame().T])
    df.loc[index, "File name"] = str(os.path.basename(abf.abfFilePath))
    df.loc[index, "Holding current"] = Ih
    df.loc[index, "Access resistance"] = Ra
    df.loc[index, "Membrane resistance"] = Rm
    df.loc[index, "Membrane capacitance"] = Cm
    df.loc[index, "Tau decay constant"] = tau

    ax.set_ylabel("I (pA)")
    ax.set_xticks(stepStart + np.arange(0, 15, 2) * abf.dataPointsPerMs)
    ax.set_xlim(stepStart - 5 * abf.dataPointsPerMs,
                stepStart + 10 * abf.dataPointsPerMs)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x - stepStart)/abf.dataPointsPerMs)))
    ax.set_xlabel("Time (ms)")

    plt.savefig(os.path.join(path_out, value + "_Graph.png"))
    plt.close()

    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")

fileName = "MemTest_Results.csv"
filePath = os.path.join(path_out, fileName)
df.to_csv(filePath)
print("All files have been processed")
