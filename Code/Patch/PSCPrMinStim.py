"""
Analyze PSCs evoked under minimal current intensity (~50% success) with paired-pulse stimulation
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


def gaussianfilter(abf, kernelSize=100, sigmaMs=5):
    assert isinstance(abf, pyabf.ABF)

    filterMatrix = np.zeros((abf.sweepCount, len(abf.sweepY)))

    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber)
        filterMatrix[sweepNumber] = abf.sweepY

    sigma = abf.dataPointsPerMs * sigmaMs
    kernel = np.arange(int(kernelSize))
    kernel = np.exp(-np.power(kernel - kernelSize / 2, 2) /
                    (2 * np.power(sigma, 2)))
    kernel /= sum(kernel)

    for index, value in enumerate(filterMatrix):
        value = np.convolve(value, kernel, mode="valid")
        nansNeeded = int((len(value) - len(value)) / 2)
        value = np.concatenate((np.full(nansNeeded, np.nan), value))
        nansNeeded = int(len(value) - len(value))
        value = np.concatenate((value, np.full(nansNeeded, np.nan)))
        filterMatrix[index] = np.concatenate((np.zeros(100), np.diff(value)))

    return filterMatrix


def clean(abf):
    assert isinstance(abf, pyabf.ABF)

    Ra = np.zeros(abf.sweepCount)
    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber)
        sweep = abf.sweepY
        Ra[sweepNumber] = np.max(
            sweep[pulseOnset: pulseOnset + postStimWindow]
        ) - np.mean(sweep[pulseOnset - preStimWindow: pulseOnset])

    Ra = 5 / Ra * 1000
    axes[0, 0].plot(Ra)

    filterMatrix = gaussianfilter(abf)

    countIPSC1 = 0
    countIPSC2 = 0
    validCount = 0
    risePos1 = np.empty(0)
    risePos2 = np.empty(0)
    amp1 = np.empty(0)
    amp2 = np.empty(0)
    peakPos1 = np.empty(0)
    peakPos2 = np.empty(0)
    cleanTrace = np.empty(0)

    for index, value in enumerate(filterMatrix):
        thresholdMin = -threshold * \
            np.std(value[stimOnset1 - preStimWindow: stimOnset1])
        peak1 = np.min(value[stimOnset1: stimOnset1 + postStimWindow])
        peak2 = np.min(value[stimOnset2: stimOnset2 + postStimWindow])

        if value[stimOnset1 - 2 * abf.dataPointsPerMs] > thresholdMin:
            validCount += 1
            abf.setSweep(index, baseline=[
                         (stimOnset1-preStimWindow)/abf.dataRate, stimOnset1/abf.dataRate])
            sweep = abf.sweepY
            if peak1 < thresholdMin:
                axes[1, 0].plot(value, color="blue", alpha=0.4)
                axes[1, 1].plot(sweep, color="blue", alpha=0.4)
                countIPSC1 += 1
                cleanTrace = np.append(cleanTrace, index)
                peakPos1 = np.append(peakPos1, np.where(
                    value[stimOnset1: stimOnset1 + postStimWindow] == peak1)[0][0])
                amp1 = np.append(sweep[int(stimOnset1 + peakPos1[-1])], amp1)
                risePos1 = np.append(
                    risePos1,
                    np.where(
                        value[stimOnset1: stimOnset1 +
                              postStimWindow] <= thresholdMin
                    )[0][0],
                )
                axes[1, 0].plot(peakPos1[-1] + stimOnset1,
                                peak1, color="green", marker="o")
                axes[1, 0].plot(risePos1[-1] + stimOnset1,
                                value[int(risePos1[-1])], color="red", marker="o")
            else:
                axes[1, 0].plot(value, color="orange", alpha=0.4)
                axes[1, 1].plot(sweep, color="orange", alpha=0.4)

            abf.setSweep(index, baseline=[
                         (stimOnset2-preStimWindow)/abf.dataRate, stimOnset2/abf.dataRate])
            sweep = abf.sweepY
            if peak2 < thresholdMin:
                abf.setSweep(index, baseline=[
                             (stimOnset2-preStimWindow)/abf.dataRate, stimOnset2/abf.dataRate])
                axes[2, 0].plot(value, color="blue", alpha=0.4)
                axes[2, 1].plot(sweep, color="blue", alpha=0.4)
                countIPSC2 += 1
                peakPos2 = np.append(peakPos2, np.where(
                    value[stimOnset2: stimOnset2 + postStimWindow] == peak2)[0][0])
                amp2 = np.append(
                    abf.sweepY[int(stimOnset2 + peakPos2[-1])], amp2)
                risePos2 = np.append(
                    risePos2,
                    np.where(
                        value[stimOnset2: stimOnset2 + postStimWindow] <= thresholdMin)[0][0])
                axes[2, 0].plot(peakPos2[-1] + stimOnset2,
                                peak2, color="green", marker="o")
                axes[2, 0].plot(risePos2[-1] + stimOnset2,
                                value[int(risePos2[-1])], color="red", marker="o")
            else:
                axes[2, 0].plot(value, color="orange", alpha=0.4)
                axes[2, 1].plot(sweep, color="orange", alpha=0.4)

    meanTrace = np.zeros(len(abf.sweepY))
    for index, value in enumerate(cleanTrace):
        abf.setSweep(
            int(value),
            baseline=[
                (stimOnset1 - preStimWindow) / abf.dataRate,
                stimOnset1 / abf.dataRate,
            ],
        )
        meanTrace += abf.sweepY

    meanTrace /= countIPSC1

    evokeRate1 = countIPSC1 / validCount
    evokeRate2 = countIPSC2 / validCount
    IPSCamp1 = -np.mean(amp1)
    IPSCamp2 = -np.mean(amp2)

    return evokeRate1, evokeRate2, IPSCamp1, IPSCamp2, meanTrace


def IPSC(abf, meanTrace):

    meanPeak = np.min(meanTrace[stimOnset1: (stimOnset1 + postStimWindow)])
    meanPeakPos = np.where(
        meanTrace[stimOnset1: (stimOnset1 + postStimWindow)] <= meanPeak
    )[0][0]

    riseOnset1 = np.where(
        meanTrace[stimOnset1: stimOnset1 +
                  meanPeakPos] >= np.min(meanTrace[stimOnset1-preStimWindow:stimOnset1])
    )[-1][-1]

    latencyIPSC1 = riseOnset1 / abf.dataPointsPerMs
    tenpercentile = meanTrace[stimOnset1 + riseOnset1] - 0.1 * \
        (meanTrace[stimOnset1 + riseOnset1] -
         meanTrace[stimOnset1 + meanPeakPos])
    ninetypercentile = meanTrace[stimOnset1 + riseOnset1] - 0.9 * (
        meanTrace[stimOnset1 + riseOnset1]-meanTrace[stimOnset1 + meanPeakPos])
    tenpercentilePos = np.where(
        meanTrace[stimOnset1: stimOnset1 + meanPeakPos] >= tenpercentile)[-1][-1]
    ninetypercentilePos = np.where(
        meanTrace[stimOnset1: stimOnset1 + meanPeakPos] >= ninetypercentile)[-1][-1]
    risetimeIPSC1 = (ninetypercentilePos - tenpercentilePos) / \
        abf.dataPointsPerMs

    axes[0, 1].plot(meanTrace)
    axes[0, 1].plot(
        riseOnset1 + stimOnset1,
        meanTrace[riseOnset1 + stimOnset1],
        "ro",
        label="Rise Onset",
    )
    axes[0, 1].plot(
        meanPeakPos + stimOnset1,
        meanTrace[meanPeakPos + stimOnset1],
        "go",
        label="IPSC peak",
    )
    axes[0, 1].plot(
        tenpercentilePos + stimOnset1,
        meanTrace[tenpercentilePos + stimOnset1],
        "bo",
        label="10th percentile",
    )
    axes[0, 1].plot(
        ninetypercentilePos + stimOnset1,
        meanTrace[ninetypercentilePos + stimOnset1],
        "bo",
        label="90th percentile",
    )

    axes[0, 1].set_ylim(meanPeak * 2.5, -meanPeak)
    axes[1, 1].set_ylim(meanPeak * 4, -meanPeak)
    axes[2, 1].set_ylim(meanPeak * 4, -meanPeak)

    return latencyIPSC1, risetimeIPSC1


def model(Pr1, Pr2):
    poissonRatio = (np.log(1-Pr2)*(Pr1))/(np.log(1-Pr1)*(Pr2))
    binomialRatio = ((1-np.sqrt(1-Pr2))*Pr1)/((1-np.sqrt(1-Pr1))*(Pr2))
    return poissonRatio, binomialRatio


path_in = r"C:\Users\isaac.chang\Desktop\Data\Patch\CsCl_Opto_Pr\PSCPrMinStim"
path_out = path_in

files = load(path_in)
print("Analyzing " + str(len(files)) + " files below:")
print(files)

df = pd.DataFrame()

for index, value in enumerate(files):
    abf = pyabf.ABF(os.path.join(path_in, value))

    threshold = 5
    preStimWindow = 5 * abf.dataPointsPerMs
    postStimWindow = 50 * abf.dataPointsPerMs
    pulseOnset = int(178.1 * abf.dataPointsPerMs)
    stimOnset1 = int(1078.1 * abf.dataPointsPerMs)
    stimOnset2 = int(1178.1 * abf.dataPointsPerMs)
    fitToFracUpper = 0.9

    fig, axes = plt.subplots(3, 2)
    fig.set_figheight(14)
    fig.set_figwidth(14)

    evokeRate1, evokeRate2, IPSCamp1, IPSCamp2, meanTrace = clean(
        abf)
    IPSClatency1, IPSCrisetime1 = IPSC(abf, meanTrace)
    poissonRatio, binomialRatio = model(evokeRate1, evokeRate2)

    df = df.append(pd.Series([], dtype="float64", name=index))
    df.loc[index, "File name"] = str(os.path.basename(abf.abfFilePath))
    df.loc[index, "evokeRate1"] = evokeRate1
    df.loc[index, "evokeRate2"] = evokeRate2
    df.loc[index, "amp1"] = IPSCamp1
    df.loc[index, "amp2"] = IPSCamp2
    df.loc[index, "Pr2/Pr1"] = evokeRate2/evokeRate1
    df.loc[index, "IPSCamp2/IPSCamp1"] = IPSCamp2/IPSCamp1
    df.loc[index, "risetime1"] = IPSCrisetime1
    df.loc[index, "latency1"] = IPSClatency1
    df.loc[index, "Poisson"] = poissonRatio
    df.loc[index, "Binomial"] = binomialRatio

    axes[0, 0].set_ylabel("Raccess (MOhm")
    axes[0, 0].set_ylim(10, 30)
    axes[0, 0].set_xlabel("Sweep number")
    axes[0, 1].set_ylabel("I (pA)")
    axes[0, 1].axvspan(stimOnset1, stimOnset1 + 2 * abf.dataPointsPerMs,
                       alpha=0.3, color="skyblue")
    axes[0, 1].set_xticks(
        stimOnset1 + np.arange(0, 500, 50) * abf.dataPointsPerMs)
    axes[0, 1].set_xlim(stimOnset1 - 10 * preStimWindow,
                        stimOnset1 + 10 * postStimWindow)
    axes[0, 1].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x - stimOnset1)/abf.dataPointsPerMs)))
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].set_ylabel("dI/dt")
    axes[1, 0].set_ylim(-2, 0.5)
    axes[1, 0].axvspan(stimOnset1, stimOnset1 + 2 * abf.dataPointsPerMs,
                       alpha=0.3, color="skyblue")
    axes[1, 0].set_xticks(
        stimOnset1 + np.arange(0, 100, 10) * abf.dataPointsPerMs)
    axes[1, 0].set_xlim(stimOnset1 - preStimWindow,
                        stimOnset1 + 2 * postStimWindow)
    axes[1, 0].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x - stimOnset1)/abf.dataPointsPerMs)))
    axes[1, 0].set_xlabel("Time (ms)")
    axes[2, 0].set_ylabel("dI/dt")
    axes[2, 0].set_ylim(-2, 0.5)
    axes[2, 0].axvspan(stimOnset2, stimOnset2 + 2 * abf.dataPointsPerMs,
                       alpha=0.3, color="skyblue")
    axes[2, 0].set_xticks(
        stimOnset2 + np.arange(0, 100, 10) * abf.dataPointsPerMs)
    axes[2, 0].set_xlim(stimOnset2 - preStimWindow,
                        stimOnset2 + 2 * postStimWindow)
    axes[2, 0].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x - stimOnset2)/abf.dataPointsPerMs)))
    axes[2, 0].set_xlabel("Time (ms)")
    axes[1, 1].set_ylabel("I (pA)")
    axes[1, 1].axvspan(stimOnset1, stimOnset1 + 2 * abf.dataPointsPerMs,
                       alpha=0.3, color="skyblue")
    axes[1, 1].set_xticks(
        stimOnset1 + np.arange(0, 100, 10) * abf.dataPointsPerMs)
    axes[1, 1].set_xlim(stimOnset1 - preStimWindow,
                        stimOnset1 + 2 * postStimWindow)
    axes[1, 1].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x - stimOnset1)/abf.dataPointsPerMs)))
    axes[1, 1].set_xlabel("Time (ms)")
    axes[2, 1].set_ylabel("I (pA)")
    axes[2, 1].axvspan(stimOnset2, stimOnset2 + 2 * abf.dataPointsPerMs,
                       alpha=0.3, color="skyblue")
    axes[2, 1].set_xticks(
        stimOnset2 + np.arange(0, 100, 10) * abf.dataPointsPerMs)
    axes[2, 1].set_xlim(stimOnset2 - preStimWindow,
                        stimOnset2 + 2 * postStimWindow)
    axes[2, 1].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x - stimOnset2)/abf.dataPointsPerMs)))
    axes[2, 1].set_xlabel("Time (ms)")

    plt.savefig(os.path.join(path_out, value + "_Graph.png"))
    plt.close()

    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")

fileName = "PSCPrMinStim_Results.csv"
filePath = os.path.join(path_out, fileName)
df.to_csv(filePath)
print("All files have been processed")
