"""
Analyze PSCs evoked using a train of 5 stimulations of different frequencies
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


def doubleexp(t, A1, tau1, A2, tau2):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)


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


def getMeanSDSweep(abf):
    assert isinstance(abf, pyabf.ABF)

    meanSweep = np.zeros(len(abf.sweepY))
    sdSweep = np.zeros(len(abf.sweepY))
    sdMatrix = np.zeros((abf.sweepCount, len(abf.sweepY)))
    risePos = np.empty(abf.sweepCount)

    baseline = [
        (stim[0] - 5 * abf.dataPointsPerMs) / abf.dataRate,
        stim[0] / abf.dataRate,
    ]

    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber, baseline=baseline)
        sweep = abf.sweepY
        sdMatrix[sweepNumber] = np.array(sweep)
        risePos[sweepNumber] = np.where(sweep[stim[0]: stim[0] + postStimWindow] >= np.min(sweep[stim[0]: stim[0] + preStimWindow])
                                        )[-1][-1]
        meanSweep += sweep

    risePos = risePos/abf.dataPointsPerMs
    jitterIPSC = np.std(risePos)
    meanSweep /= abf.sweepCount
    sdSweep = sdMatrix.std(0)

    return meanSweep, sdSweep, jitterIPSC


def IPSCHz(
    abf,
    meanTrace,
    sdTrace,
):

    assert isinstance(abf, pyabf.ABF)

    ampIPSC = np.empty(5)
    sdIPSC = np.empty(5)
    ampIPSCPos = np.empty(5)
    axes[0].plot(meanTrace)
    axes[0].plot(sdTrace)

    for index, value in enumerate(stim):
        meanPeak = np.min(meanTrace[value: (value + postStimWindow)])
        meanBsl = np.mean(meanTrace[(value - preStimWindow): value])
        sdPeak = np.max(sdTrace[value: (value + postStimWindow)])

        ampIPSCPos[index] = np.where(
            meanTrace[value: (value + postStimWindow)] == meanPeak
        )[0][0]
        ampIPSC[index] = -(meanPeak - meanBsl)
        sdIPSC[index] = sdPeak
        axes[0].plot(
            value + ampIPSCPos[index],
            meanTrace[int(value + ampIPSCPos[index])],
            "go",
        )
        axes[0].axvspan(stim[index], stim[index] + 2 * abf.dataPointsPerMs,
                        alpha=0.3, color="skyblue")

    meanPeakPos = int(ampIPSCPos[0])
    riseOnset = np.where(
        meanTrace[stim[0]: stim[0] + meanPeakPos]
        >= np.min(meanTrace[stim[0]-preStimWindow: stim[0]])
    )[-1][-1]
    axes[0].plot(
        riseOnset + stim[0],
        meanTrace[riseOnset + stim[0]],
        "ro",
    )
    latencyIPSC = riseOnset / abf.dataPointsPerMs
    tenpercentile = meanTrace[stim[0] + riseOnset] - 0.1 * \
        (meanTrace[stim[0] + riseOnset]-meanTrace[stim[0] + meanPeakPos])
    ninetypercentile = meanTrace[stim[0] + riseOnset] - 0.9 * \
        (meanTrace[stim[0] + riseOnset]-meanTrace[stim[0] + meanPeakPos])
    tenpercentilePos = np.where(
        meanTrace[stim[0]: stim[0] + meanPeakPos] >= tenpercentile)[-1][-1]
    ninetypercentilePos = np.where(
        meanTrace[stim[0]: stim[0] + meanPeakPos] >= ninetypercentile)[-1][-1]
    risetimeIPSC = (ninetypercentilePos - tenpercentilePos) / \
        abf.dataPointsPerMs

    traceXOffset = int(stim[4] + meanPeakPos)
    fitThis = meanTrace[traceXOffset:]
    zeroI = np.where(fitThis >= 0)[0]
    if len(zeroI):
        zeroI = zeroI[0]
        fitThis = fitThis[:zeroI]
        chargeTrace = meanTrace[stim[0]+riseOnset:traceXOffset+zeroI]
    else:
        chargeTrace = meanTrace[stim[0]+riseOnset:]
    upperFracVal = fitToFracUpper * fitThis[0]
    upperI = np.where(fitThis < upperFracVal)[0][0]
    fitThis = fitThis[upperI:]
    p0 = [-100, 2000, -100, 10000]
    [A1, tau1, A2, tau2], cv = scipy.optimize.curve_fit(
        doubleexp, np.arange(len(fitThis)), fitThis, p0=p0, maxfev=10000
    )
    axes[0].plot(
        np.arange(len(fitThis)) + traceXOffset,
        doubleexp(np.arange(len(fitThis)), A1, tau1, A2, tau2),
        "--",
        color="red",
    )
    axes[0].fill_between(np.arange(stim[0]+riseOnset, stim[0]+riseOnset + len(
        chargeTrace)), chargeTrace, 0, where=(chargeTrace < 0), color='blue', alpha=0.5)
    axes[0].set_ylim(meanPeak * 1.5, -meanPeak)
    tauIPSC = ((A1 * tau1 + A2 * tau2) / (A1 + A2)) / abf.dataPointsPerMs
    chargeIPSC = -np.sum(chargeTrace) / abf.dataPointsPerMs / 1000

    return (
        latencyIPSC,
        ampIPSC,
        risetimeIPSC,
        chargeIPSC,
        tauIPSC,
        sdIPSC,
    )


def IPSCHz2(
    abf,
    filterMatrix,
):

    assert isinstance(abf, pyabf.ABF)

    countIPSC = np.zeros(len(stim))
    peakPos = np.empty((abf.sweepCount, len(stim)))

    for index, value in enumerate(stim):
        for index2, value2 in enumerate(filterMatrix):
            peak = np.min(value2[stim[index]: int(
                stim[index] + postStimWindow)])
            peakPos[index2][index] = np.where(value2 == peak)[0][0]
            axes[1].plot(value2, color="blue", alpha=0.5)
            if peak <= threshold:
                countIPSC[index] += 1
                axes[1].plot(peakPos[index2][index], peak,
                             color="green", marker="o")
        axes[1].axvspan(stim[index], stim[index] + 2 * abf.dataPointsPerMs,
                        alpha=0.3, color="skyblue")
        axes[1].set_xticks(stim[0] + np.arange(0, stim[4]-stim[0] + 50 *
                           abf.dataPointsPerMs, (stim[4]-stim[0] + 50 * abf.dataPointsPerMs)/5))
        axes[1].set_xlim(stim[0] - preStimWindow,
                         stim[4] + 50 * abf.dataPointsPerMs)

    evokeRate = countIPSC/abf.sweepCount

    return evokeRate


path_in = r"C:\Users\isaac.chang\Desktop\Data\Patch\CsCl_Opto\PSCHz\50Hz"
path_out = path_in

files = load(path_in)
print("Analyzing " + str(len(files)) + " files below:")
print(files)

df = pd.DataFrame()

stim2 = [11250, 16250, 21250, 26250, 31250]
stim5 = [11250, 13250, 15250, 17250, 19250]
stim10 = [11250, 12250, 13250, 14250, 15250]
stim20 = [11250, 11750, 12250, 12750, 13250]
stim30 = [11250, 11553, 11856, 12159, 12462]
stim50 = [11250, 11450, 11650, 11850, 12050]

for index, value in enumerate(files):
    abf = pyabf.ABF(os.path.join(path_in, value))

    stim = stim50
    threshold = -0.2
    preStimWindow = 5 * abf.dataPointsPerMs
    postStimWindow = 20 * abf.dataPointsPerMs
    fitToFracUpper = 0.9

    fig, axes = plt.subplots(1, 4)
    fig.set_figheight(5)
    fig.set_figwidth(20)

    meanTrace, sdTrace, jitterIPSC = getMeanSDSweep(abf)

    (
        latencyIPSC,
        ampIPSC,
        risetimeIPSC,
        chargeIPSC,
        tauIPSC,
        sdIPSC,
    ) = IPSCHz(
        abf,
        meanTrace,
        sdTrace,
    )

    normAmpIPSC = ampIPSC / ampIPSC[0]
    filterMatrix = gaussianfilter(abf)
    evokeRate = IPSCHz2(abf, filterMatrix)

    axes[2].plot(normAmpIPSC)
    axes[3].plot(evokeRate)

    axes[0].set_ylabel("I (pA)")

    axes[0].set_xticks(stim[0] + np.arange(0, stim[4]-stim[0] + 1000 *
                       abf.dataPointsPerMs, (stim[4]-stim[0] + 1000 * abf.dataPointsPerMs)/5))
    axes[0].set_xlim(stim[0] - preStimWindow, stim[4] +
                     1000 * abf.dataPointsPerMs)
    axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x - stim[0])/abf.dataPointsPerMs)))
    axes[0].set_xlabel("Time (ms)")
    axes[1].set_ylabel("dI/dt")
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x - stim[0])/abf.dataPointsPerMs)))
    axes[1].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Norm IPSC amp")
    axes[2].axhline(1, ls=":")
    axes[2].set_xlabel("Stimulation number")
    axes[3].set_ylabel("Evoke Rate")
    axes[3].set_ylim(0, 1.1)
    axes[3].set_xlabel("Stimulation number")
    axes[3].axhline(1, ls=":")

    df = df.append(pd.Series([], dtype="float64", name=index))
    df.loc[index, "File name"] = str(os.path.basename(abf.abfFilePath))
    df.loc[index, "Latency"] = latencyIPSC
    for index2, value2 in enumerate(ampIPSC):
        df.loc[index, str(index2) + " amplitude"] = value2
    for index2, value2 in enumerate(normAmpIPSC):
        df.loc[index, str(index2) + " normalized amplitude"] = value2
    for index2, value2 in enumerate(evokeRate):
        df.loc[index, str(index2) + " success rate"] = value2
    df.loc[index, "Train tau"] = tauIPSC
    df.loc[index, "Train charge"] = chargeIPSC

    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")
    plt.savefig(
        os.path.join(path_out, str(
            os.path.basename(abf.abfFilePath) + "_Graph.png"))
    )
    plt.close()

fileName = "PSCHz_Results.csv"
filePath = os.path.join(path_out, fileName)
df.to_csv(filePath)
print("All files have been processed")
