"""
Analzye PSC evoked under different command potentials, for nRT perforated-patch recordings 
Author: Isaac Chang
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import pyabf
from scipy.signal import medfilt


def load(path_in):
    files = []
    for i in os.listdir(path_in):
        if os.path.splitext(i)[-1].lower() == ".abf":
            files.append(i)
    files.sort()
    return files


def stats(x, y, bins):
    x = np.array(x)
    y = np.array(y)

    binidx = np.digitize(x, bins) - 1

    means = []
    sems = []

    for i in range(len(bins) - 1):
        vals = y[binidx == i]
        if len(vals) > 0:
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
        else:
            means.append(np.nan)
            sems.append(np.nan)
    centers = (bins[:-1] + bins[1:]) / 2

    return centers, np.array(means), np.array(sems)


def IVCurve(abf):

    assert isinstance(abf, pyabf.ABF)
    IPSCamp = np.zeros(len(abf.sweepList))
    Ihamp = np.zeros(len(abf.sweepList))
    VStepCor = np.array(VStep.copy())
    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber)
        sweep = abf.sweepY
        Ih = np.mean(sweep[stimOnset - preStimWindow: stimOnset])
        Ihamp[sweepNumber] = Ih
        VStepCor[sweepNumber] = VStep[sweepNumber] - (Ih) * Ra[index] / 1000
        sweep = sweep - Ih
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

    Vm2 = np.where(Ihamp >= 0)[0][0]
    Vmy = Ihamp[Vm2] - Ihamp[Vm2 - 1]
    Vmx = VStepCor[Vm2 - 1] + (Ihamp[Vm2 - 1] / Vmy) * (
        VStepCor[Vm2 - 1] - VStepCor[Vm2]
    )

    axes[1].plot(VStepCor, Ihamp)
    axes[1].set_ylim(np.min(Ihamp), np.max(Ihamp))
    axes[1].axvline(x=Vmx, color="Blue", linestyle="--")
    axes[1].annotate(
        f"Vm = {Vmx:.2f}",
        (Vmx, 0),
        textcoords="offset points",
        xytext=(0, 10),
        ha="left",
    )

    idx1 = len(IPSCamp)
    for i in range(len(IPSCamp) - 1):
        change = abs(IPSCamp[i + 1] - IPSCamp[i])
        if change > 3 * abs(IPSCamp[i]):
            idx1 = i + 1
            break
    idx2 = np.where(np.array(VStepCor) <= -20)[0][-1] + 1
    idx = min(idx1, idx2)

    VStepCor = VStepCor[:idx]
    IPSCamp = IPSCamp[:idx]
    slope, intercept = np.polyfit(VStepCor, IPSCamp, 1)

    x = np.linspace(-120, 0, 100)
    y = slope * x + intercept

    EClx = -intercept/slope

    axes[2].plot(x, y, '-')
    axes[2].plot(VStepCor, IPSCamp, '-o')
    axes[2].axvline(x=EClx, color="Blue", linestyle="--")
    axes[2].annotate(
        f"ECl = {EClx:.2f}",
        (EClx, 0),
        textcoords="offset points",
        xytext=(0, 10),
        ha="left",
    )

    return EClx, Vmx


path_in = r"\\hive\Paz-Lab\Isaac\GPe\Data\Patch\KCl_Opto\IVCurveNRTPP"
path_out = path_in
pathinfo = path_in

files = load(path_in)
info = pd.read_csv(os.path.join(pathinfo, "info.csv"))
Ra = info["Ra"].tolist()
response = info["response"].tolist()


df = pd.DataFrame()

print("Analyzing " + str(len(files)) + " files below:")
print(files)

for index, value in enumerate(files):
    abf = pyabf.ABF(os.path.join(path_in, value))

    stimOnset = int(531.2 * abf.dataPointsPerMs)
    preStimWindow = int(5 * abf.dataPointsPerMs)
    postStimWindow = int(20 * abf.dataPointsPerMs)
    junctionPotential = 0
    VStep = [
        i - junctionPotential for i in range(-130, (-130 + abf.sweepCount * 5), 5)]

    fig, axes = plt.subplots(1, 3)
    fig.set_figheight(4)
    fig.set_figwidth(12)

    EClx, Vmx = IVCurve(abf)

    df = df.append(pd.Series([], dtype="float64", name=index))
    df.loc[index, "File name"] = str(os.path.basename(abf.abfFilePath))
    df.loc[index, "Chloride reversal potential"] = EClx
    df.loc[index, "Membrane potential"] = Vmx

    axes[0].set_ylabel("I (pA)")
    axes[0].axvspan(0, 2 * abf.dataPointsPerMs,
                    alpha=0.3, color="skyblue")
    axes[0].set_xticks(np.arange(0, 20, 5) * abf.dataPointsPerMs)
    axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: float(x/abf.dataPointsPerMs)))
    axes[1].set_ylabel("Ih (pA)")
    axes[1].axhline(y=0, color="gray", linestyle="--")
    axes[1].axvline(x=0, color="gray", linestyle="--")
    axes[1].set_xlabel("Vm (mV)")
    axes[1].set_xlim(-120, -20)
    axes[2].set_ylabel("Ih (pA)")
    axes[2].axhline(y=0, color="gray", linestyle="--")
    axes[2].axvline(x=0, color="gray", linestyle="--")
    axes[2].set_xlabel("Vm (mV)")
    axes[2].set_xlim(-120, -20)

    plt.savefig(os.path.join(path_out, value + "_Graph.png"))
    plt.close()

    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")

fileName = "IVCurveNRTPP_Results.csv"
filePath = os.path.join(path_out, fileName)
df.to_csv(filePath)
print("All files have been processed")
