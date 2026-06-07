"""
Analyze PSCs evoked under minimal current intensity (~50% success)
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
    ax1.plot(Ra)

    filterMatrix = gaussianfilter(abf)

    countIPSC = 0
    validCount = 0
    risePos = np.empty(0)
    cleanTrace = np.empty(0)

    for index, value in enumerate(filterMatrix):
        thresholdMin = -threshold * \
            np.std(value[stimOnset - preStimWindow: stimOnset])
        peak = np.min(value[stimOnset: stimOnset + postStimWindow])
        peakPos = np.where(
            value[stimOnset: stimOnset + postStimWindow] == peak)[0][0]

        if value[stimOnset - 2 * abf.dataPointsPerMs] > thresholdMin:
            validCount += 1

            if peak < thresholdMin:
                ax2.plot(value, color="blue", alpha=0.4)
                countIPSC += 1
                cleanTrace = np.append(cleanTrace, index)
                risePos = np.append(
                    risePos,
                    np.where(
                        value[stimOnset: stimOnset +
                              postStimWindow] <= thresholdMin
                    )[0][0],
                )
                ax2.plot(peakPos + stimOnset, peak, color="green", marker="o")
                ax2.plot(risePos[-1] + stimOnset,
                         value[int(risePos[-1])], color="red", marker="o")
            else:
                ax2.plot(value, color="orange", alpha=0.4)

    evokeRate = countIPSC / validCount

    meanTrace = np.zeros(len(abf.sweepY))
    sdMatrix = np.zeros((len(cleanTrace), len(abf.sweepY)))
    sdTrace = np.zeros(len(abf.sweepY))

    for index, value in enumerate(cleanTrace):
        abf.setSweep(
            int(value),
            baseline=[
                (stimOnset - preStimWindow) / abf.dataRate,
                stimOnset / abf.dataRate,
            ],
        )
        sdMatrix[index] = np.array(abf.sweepY)
        meanTrace += abf.sweepY

    meanTrace /= countIPSC
    sdTrace = sdMatrix.std(0)

    ax3.plot(meanTrace)
    ax3.plot(sdTrace)

    return evokeRate, meanTrace, sdTrace


def IPSC(abf, meanTrace, sdTrace):
    meanPeak = np.min(meanTrace[stimOnset: (stimOnset + postStimWindow)])
    meanPeakPos = np.where(
        meanTrace[stimOnset: (stimOnset + postStimWindow)] <= meanPeak
    )[0][0]

    riseOnset = np.where(
        meanTrace[stimOnset: stimOnset +
                  meanPeakPos] >= np.min(meanTrace[stimOnset-preStimWindow:stimOnset])
    )[-1][-1]
    sdPeak = np.max(sdTrace[stimOnset: (stimOnset + postStimWindow)])

    latencyIPSC = riseOnset / abf.dataPointsPerMs
    ampIPSC = -meanPeak
    tenpercentile = meanTrace[stimOnset + riseOnset] - 0.1 * \
        (meanTrace[stimOnset + riseOnset]-meanTrace[stimOnset + meanPeakPos])
    ninetypercentile = meanTrace[stimOnset + riseOnset] - 0.9 * \
        (meanTrace[stimOnset + riseOnset]-meanTrace[stimOnset + meanPeakPos])
    tenpercentilePos = np.where(
        meanTrace[stimOnset: stimOnset + meanPeakPos] >= tenpercentile)[-1][-1]
    ninetypercentilePos = np.where(
        meanTrace[stimOnset: stimOnset + meanPeakPos] >= ninetypercentile)[-1][-1]
    risetimeIPSC = (ninetypercentilePos - tenpercentilePos) / \
        abf.dataPointsPerMs
    sdIPSC = sdPeak

    ax3.plot(meanTrace)
    ax3.plot(
        riseOnset + stimOnset,
        meanTrace[riseOnset + stimOnset],
        "ro",
    )
    ax3.plot(
        meanPeakPos + stimOnset,
        meanTrace[meanPeakPos + stimOnset],
        "go",
    )
    ax3.plot(
        tenpercentilePos + stimOnset,
        meanTrace[tenpercentilePos + stimOnset],
        "bo",
    )
    ax3.plot(
        ninetypercentilePos + stimOnset,
        meanTrace[ninetypercentilePos + stimOnset],
        "bo",
    )

    traceXOffset = stimOnset + meanPeakPos
    fitThis = meanTrace[traceXOffset:]
    zeroI = np.where(fitThis >= 0)[0]
    if len(zeroI):
        zeroI = zeroI[0]
        fitThis = fitThis[:zeroI]
    chargeTrace = meanTrace[stimOnset+riseOnset:traceXOffset+zeroI]
    upperFracVal = fitToFracUpper * fitThis[0]
    upperI = np.where(fitThis < upperFracVal)[0][0]
    fitThis = fitThis[upperI:]
    p0 = [-30, 5000, -30, 500]
    [A1, tau1, A2, tau2], cv = scipy.optimize.curve_fit(
        doubleexp, np.arange(len(fitThis)), fitThis, p0=p0
    )
    ax3.plot(
        np.arange(len(fitThis)) + traceXOffset,
        doubleexp(np.arange(len(fitThis)), A1, tau1, A2, tau2),
        "--",
        color="red",
    )
    ax3.fill_between(np.arange(stimOnset+riseOnset, traceXOffset+zeroI),
                     chargeTrace, 0, where=(chargeTrace < 0), color='blue', alpha=0.5)
    ax3.set_ylim(meanPeak * 1.5, -meanPeak)
    tauIPSC = ((A1 * tau1 + A2 * tau2) / (A1 + A2)) / abf.dataPointsPerMs
    chargeIPSC = -np.sum(chargeTrace) / abf.dataPointsPerMs/1000

    return (
        latencyIPSC,
        ampIPSC,
        risetimeIPSC,
        chargeIPSC,
        tauIPSC,
        sdIPSC,
    )


path_in = r"C:\Users\isaac.chang\Desktop\Data\Patch\CsCl_Opto\PSCSingleMinStim"
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
    pulseOnset = 225 * abf.dataPointsPerMs
    stimOnset = 1125 * abf.dataPointsPerMs
    fitToFracUpper = 0.9

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.set_figheight(14)
    fig.set_figwidth(7)

    evokeRate, meanTrace, sdTrace = clean(abf)

    (
        latencyIPSC,
        ampIPSC,
        risetimeIPSC,
        chargeIPSC,
        tauIPSC,
        sdIPSC,
    ) = IPSC(abf, meanTrace, sdTrace)

    df = df.append(pd.Series([], dtype="float64", name=index))
    df.loc[index, "File name"] = str(os.path.basename(abf.abfFilePath))
    df.loc[index, "Latency"] = latencyIPSC
    df.loc[index, "Amplitude"] = ampIPSC
    df.loc[index, "Rise time"] = risetimeIPSC
    df.loc[index, "Charge"] = chargeIPSC
    df.loc[index, "Tau"] = tauIPSC
    df.loc[index, "CV"] = sdIPSC/ampIPSC
    df.loc[index, "Evoke Rate"] = evokeRate

    ax1.set_ylabel("Raccess (MΩ)")
    ax1.set_ylim(10, 30)
    ax1.set_xlabel("Sweep number")
    ax2.set_ylabel("dI/dt")
    ax2.axvspan(stimOnset, stimOnset + 2 * abf.dataPointsPerMs,
                alpha=0.3, color="skyblue")
    ax2.set_xticks(stimOnset + np.arange(0, 100, 10) * abf.dataPointsPerMs)
    ax2.set_xlim(stimOnset - preStimWindow, stimOnset + 2 * postStimWindow)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x - stimOnset)/abf.dataPointsPerMs)))
    ax2.set_xlabel("Time (ms)")
    ax3.set_ylabel("I (pA)")
    ax3.axvspan(stimOnset, stimOnset + 2 * abf.dataPointsPerMs,
                alpha=0.3, color="skyblue")
    ax3.set_xticks(stimOnset + np.arange(0, 500, 50) * abf.dataPointsPerMs)
    ax3.set_xlim(stimOnset - 10 * preStimWindow,
                 stimOnset + 10 * postStimWindow)
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: int((x - stimOnset)/abf.dataPointsPerMs)))
    ax3.set_xlabel("Time (ms)")

    plt.savefig(os.path.join(path_out, value + "_Graph.png"))
    plt.close()

    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")

fileName = "PSCSingleMinStim_Results.csv"
filePath = os.path.join(path_out, fileName)
df.to_csv(filePath)
print("All files have been processed")
