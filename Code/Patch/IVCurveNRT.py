"""
Analzye PSC evoked under different command potentials, for nRT recordings 
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


def IVCurve(abf, VStep):
    assert isinstance(abf, pyabf.ABF)
    IPSCamp = np.zeros(len(abf.sweepList))
    for sweepNumber in abf.sweepList:
        baseline = [
            (stimOnset - preStimWindow) / abf.dataRate,
            stimOnset / abf.dataRate,
        ]
        abf.setSweep(sweepNumber, baseline=baseline)
        sweep = abf.sweepY
        posPeak = np.max(sweep[stimOnset: stimOnset + postStimWindow])
        negPeak = np.min(sweep[stimOnset: stimOnset + postStimWindow])
        if posPeak > abs(negPeak):
            peak = posPeak
        else:
            peak = negPeak
        peakPos = np.where(
            sweep[stimOnset: stimOnset + postStimWindow] == peak)[0][0]
        IPSCamp[sweepNumber] = peak
        axes[0].plot(
            sweep[stimOnset: stimOnset + postStimWindow], color="blue", alpha=0.5
        )
        axes[0].plot(peakPos, peak, color="green", marker="o")

        if sweepNumber == 0:
            riseOnset = np.where(
                sweep[stimOnset: stimOnset + peakPos]
                >= np.min(sweep[stimOnset: stimOnset + 1 * abf.dataPointsPerMs])
            )[-1][-1]
            axes[0].axvline(riseOnset, ls=":")
            latency = riseOnset / abf.dataPointsPerMs

    if len(np.where(np.isclose(VStep, VStepEnd))[0]) > 0:
        idx = min(abf.sweepCount, np.where(
            np.isclose(VStep, VStepEnd))[0][0] + 1)
    else:
        idx = abf.sweepCount
    VStep = VStep[:idx]
    IPSCamp = IPSCamp[:idx]
    ECl2 = np.where(IPSCamp >= 0)[0][0]
    ECly = IPSCamp[ECl2] - IPSCamp[ECl2 - 1]
    EClx = VStep[ECl2 - 1] + (IPSCamp[ECl2 - 1] / ECly) * (
        VStep[ECl2 - 1] - VStep[ECl2]
    )
    axes[1].plot(VStep, IPSCamp)
    axes[1].set_ylim(np.min(IPSCamp), np.max(IPSCamp))
    axes[1].axvline(x=EClx, color="Blue", linestyle="--")
    axes[1].annotate(
        "ECl = " + str(EClx),
        (EClx, 0),
        textcoords="offset points",
        xytext=(0, 10),
        ha="left",
    )

    return IPSCamp, VStep, EClx, latency


path_in = r"C:\Users\isaac.chang\Desktop\Data\Patch\Kgluc_Opto\nRT\IVCurveNRT"
path_out = path_in

files = load(path_in)

df = pd.DataFrame()

print("Analyzing " + str(len(files)) + " files below:")
print(files)

for index, value in enumerate(files):
    abf = pyabf.ABF(os.path.join(path_in, value))

    stimOnset = int(529.9 * abf.dataPointsPerMs)
    preStimWindow = int(5 * abf.dataPointsPerMs)
    postStimWindow = int(20 * abf.dataPointsPerMs)
    junctionPotential = 15
    VStep = [i - junctionPotential for i in range(-90, 20, 5)]
    VStepEnd = -20

    fig, axes = plt.subplots(1, 2)
    fig.set_figheight(4)
    fig.set_figwidth(10)

    IPSCamp, VStep, EClx, latency = IVCurve(abf, VStep)

    df = df.append(pd.Series([], dtype="float64", name=index))
    df.loc[index, "File name"] = str(os.path.basename(abf.abfFilePath))
    df.loc[index, "Latency"] = latency
    df.loc[index, "ECl"] = EClx
    for index2, value2 in enumerate(VStep):
        df.loc[index, value2] = IPSCamp[index2]

    axes[0].set_ylabel("I (pA)")
    axes[0].axvspan(0, 2 * abf.dataPointsPerMs,
                    alpha=0.3, color="skyblue")
    axes[0].set_xticks(np.arange(0, 20, 5) * abf.dataPointsPerMs)
    axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: float(x/abf.dataPointsPerMs)))
    axes[1].set_ylabel("I (pA)")
    axes[1].axhline(y=0, color="gray", linestyle="--")
    axes[1].axvline(x=0, color="gray", linestyle="--")
    axes[1].set_xlabel("Vm (mV)")
    axes[1].set_xlim(-110, -10)
    plt.savefig(os.path.join(path_out, value + "_Graph.png"))
    plt.close()

    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")

fileName = "IVCurveNRT_Results.csv"
filePath = os.path.join(path_out, fileName)
df.to_csv(filePath)
print("All files have been processed")
