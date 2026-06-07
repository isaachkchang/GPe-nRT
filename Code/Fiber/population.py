"""
Analzye CSV files containing z-scored GCaMP signals and movement data, for population data.
Author: Isaac Chang
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def loadfiles():
    files = []
    for i in os.listdir(pathfiles):
        if os.path.splitext(i)[-1].lower() == ".csv":
            files.append(i)
    files.sort()
    return files


def correlation(data1, data2, lag):
    mask = ~np.isnan(data1) & ~np.isnan(data2)
    data1 = data1[mask]
    data2 = data2[mask]
    corr = np.corrcoef(data1, np.roll(data2, lag))[0, 1]
    return corr


pathfiles = r"\\hive\Paz-Lab\Isaac\GPe\Data\Fiber\Analysis"
files = loadfiles()

cortimeWindow = range(-50, 50)
bouttimeWindow = 50
timeWindow = np.arange(-5, 5, 0.1)
onoffsetThreshold = 2
mvtThreshold = 5
angVelThreshold = np.pi/2

print("Analyzing " + str(len(files)) + " files below:")
print(files)

allmvtcorrelations = []
allturncorrelations = []
allinitiationMvtSegments = []
allinitiationGCaMPSegments = []
allterminationMvtSegments = []
allterminationGCaMPSegments = []
allinitiationleftMvtSegments = []
allinitiationleftGCaMPSegments = []
allterminationleftMvtSegments = []
allterminationleftGCaMPSegments = []
allinitiationrightMvtSegments = []
allinitiationrightGCaMPSegments = []
allterminationrightMvtSegments = []
allterminationrightGCaMPSegments = []
allallinitiationMvtSegments = []
allallinitiationGCaMPSegments = []
allallterminationMvtSegments = []
allallterminationGCaMPSegments = []
allallinitiationleftMvtSegments = []
allallinitiationleftGCaMPSegments = []
allallterminationleftMvtSegments = []
allallterminationrightMvtSegments = []
allallinitiationrightMvtSegments = []
allallinitiationrightGCaMPSegments = []
allallterminationleftGCaMPSegments = []
allallterminationrightGCaMPSegments = []

outdf = pd.DataFrame()

fig1, axes1 = plt.subplots(2, 8, figsize=(25, 10), gridspec_kw={
    'height_ratios': [1, 3]})

for index, value in enumerate(files):

    df = pd.read_csv(os.path.join(pathfiles, value))
    GCaMP = df["GCaMP"].to_numpy()
    speed = df["Speed"].to_numpy()
    speed = np.convolve(speed, np.ones(15)/15, mode="same")
    heading = df["Heading"].to_numpy()
    angVel = df["Angular Velocity"].to_numpy()
    angVel = savgol_filter(angVel, 5, 3)
    angVel = -angVel
    GCaMP = GCaMP[:-2]
    speed = speed[:-2]
    angVel = angVel[:-2]
    piyticks = np.arange(-4, 4.5, 1) * np.pi

    mvtcutoff = speed > onoffsetThreshold
    onsetmvt = np.where((mvtcutoff[1:] & ~mvtcutoff[:-1]))[0] + 1
    offsetmvt = np.where((~mvtcutoff[1:] & mvtcutoff[:-1]))[0] + 1
    if onsetmvt.size != offsetmvt.size:
        offsetmvt = offsetmvt[1:] 
    onsetbout = []
    offsetbout = []
    for index2, value2 in enumerate(onsetmvt):
        if index2 < offsetmvt.size:
            if np.max(speed[value2: offsetmvt[index2]]) > mvtThreshold:
                onsetbout.append(onsetmvt[index2])
                offsetbout.append(offsetmvt[index2])

    leftmask = angVel < -angVelThreshold
    rightmask = angVel > angVelThreshold
    onsetleft = np.where(np.diff(leftmask.astype(int)) == 1)[0] + 1
    offsetleft = np.where(np.diff(leftmask.astype(int)) == -1)[0] + 1
    onsetright = np.where(np.diff(rightmask.astype(int)) == 1)[0] + 1
    offsetright = np.where(np.diff(rightmask.astype(int)) == -1)[0] + 1

    mask = (speed >= 0) & (speed <= 1)
    mvtcorrelations = [correlation(
        speed[mask], GCaMP[mask], lag) for lag in cortimeWindow]
    allmvtcorrelations.append(mvtcorrelations)

    turncorrelations = [correlation(
        angVel, GCaMP, lag) for lag in cortimeWindow]
    allturncorrelations.append(turncorrelations)

    initiationMvtSegments = []
    initiationGCaMPSegments = []
    terminationMvtSegments = []
    terminationGCaMPSegments = []
    initiationleftMvtSegments = []
    initiationleftGCaMPSegments = []
    terminationleftGCaMPSegments = []
    terminationleftMvtSegments = []
    initiationrightMvtSegments = []
    initiationrightGCaMPSegments = []
    terminationrightMvtSegments = []
    terminationrightGCaMPSegments = []

    for n in onsetbout:
        if n > bouttimeWindow:
            start = int(n - bouttimeWindow)
            end = start + 100
            if end < len(speed):
                initiationMvtSegments.append(speed[start:end])
                initiationGCaMPSegments.append(GCaMP[start:end])
                allallinitiationMvtSegments.append(speed[start:end])
                allallinitiationGCaMPSegments.append(GCaMP[start:end])
            else:
                break
    for n in offsetbout:
        if n > bouttimeWindow:
            start = int(n - bouttimeWindow)
            end = start + 100
            if end < len(speed):
                terminationMvtSegments.append(speed[start:end])
                terminationGCaMPSegments.append(GCaMP[start:end])
                allallterminationMvtSegments.append(speed[start:end])
                allallterminationGCaMPSegments.append(GCaMP[start:end])
            else:
                break
    for n in onsetleft:
        if n > bouttimeWindow:
            start = int(n - bouttimeWindow)
            end = start + 100
            if end < len(angVel):
                initiationleftMvtSegments.append(angVel[start:end])
                initiationleftGCaMPSegments.append(GCaMP[start:end])
                allallinitiationleftMvtSegments.append(angVel[start:end])
                allallinitiationleftGCaMPSegments.append(GCaMP[start:end])
            else:
                break
    for n in offsetleft:
        if n > bouttimeWindow:
            start = int(n - bouttimeWindow)
            end = start + 100
            if end < len(angVel):
                terminationleftMvtSegments.append(angVel[start:end])
                terminationleftGCaMPSegments.append(GCaMP[start:end])
                allallterminationleftMvtSegments.append(angVel[start:end])
                allallterminationleftGCaMPSegments.append(GCaMP[start:end])
            else:
                break
    for n in onsetright:
        if n > bouttimeWindow:
            start = int(n - bouttimeWindow)
            end = start + 100
            if end < len(angVel):
                initiationrightMvtSegments.append(angVel[start:end])
                initiationrightGCaMPSegments.append(GCaMP[start:end])
                allallinitiationrightMvtSegments.append(angVel[start:end])
                allallinitiationrightGCaMPSegments.append(GCaMP[start:end])
            else:
                break
    for n in offsetright:
        if n > bouttimeWindow:
            start = int(n - bouttimeWindow)
            end = start + 100
            if end < len(angVel):
                terminationrightMvtSegments.append(angVel[start:end])
                terminationrightGCaMPSegments.append(GCaMP[start:end])
                allallterminationrightMvtSegments.append(angVel[start:end])
                allallterminationrightGCaMPSegments.append(GCaMP[start:end])
            else:
                break

    allinitiationGCaMPSegments.append(np.mean(initiationGCaMPSegments, axis=0))
    allinitiationMvtSegments.append(np.mean(initiationMvtSegments, axis=0))
    allterminationGCaMPSegments.append(
        np.mean(terminationGCaMPSegments, axis=0))
    allterminationMvtSegments.append(np.mean(terminationMvtSegments, axis=0))
    allinitiationleftGCaMPSegments.append(
        np.mean(initiationleftGCaMPSegments, axis=0))
    allinitiationleftMvtSegments.append(
        np.mean(initiationleftMvtSegments, axis=0))
    allterminationleftGCaMPSegments.append(
        np.mean(terminationleftGCaMPSegments, axis=0))
    allterminationleftMvtSegments.append(
        np.mean(terminationleftMvtSegments, axis=0))
    allinitiationrightGCaMPSegments.append(
        np.mean(initiationrightGCaMPSegments, axis=0))
    allinitiationrightMvtSegments.append(
        np.mean(initiationrightMvtSegments, axis=0))
    allterminationrightGCaMPSegments.append(
        np.mean(terminationrightGCaMPSegments, axis=0))
    allterminationrightMvtSegments.append(
        np.mean(terminationrightMvtSegments, axis=0))

    print(value + " has been processed")

meanmvtcorrelations = np.mean(allmvtcorrelations, axis=0)
print(np.max(meanmvtcorrelations))
semmvtcorrelations = np.std(
    allmvtcorrelations, axis=0, ddof=1)/np.sqrt(len(files))
print(semmvtcorrelations[np.argmax(meanmvtcorrelations)])

meanturncorrelations = np.mean(allturncorrelations, axis=0)
semturncorrelations = np.std(
    allturncorrelations, axis=0, ddof=1)/np.sqrt(len(files))

segments = {
    "init": {
        "gcamp": allinitiationGCaMPSegments,
        "move": allinitiationMvtSegments,
        "gcampleft": allinitiationleftGCaMPSegments,
        "moveleft": allinitiationleftMvtSegments,
        "gcampright": allinitiationrightGCaMPSegments,
        "moveright": allinitiationrightMvtSegments
    },
    "term": {
        "gcamp": allterminationGCaMPSegments,
        "move": allterminationMvtSegments,
        "gcampleft": allterminationleftGCaMPSegments,
        "moveleft": allterminationleftMvtSegments,
        "gcampright": allterminationrightGCaMPSegments,
        "moveright": allterminationrightMvtSegments
    }
}

means = {}
sems = {}

for key in segments:
    means[key] = {}
    sems[key] = {}
    for typ in ["gcamp", "move", "gcampleft", "moveleft", "gcampright", "moveright"]:
        data = np.array(segments[key][typ])
        means[key][typ] = np.mean(data, axis=0)
        sems[key][typ] = np.std(data, axis=0, ddof=1) / np.sqrt(len(files))

axes1[0, 6].plot(timeWindow, meanmvtcorrelations)
axes1[0, 6].fill_between(timeWindow, meanmvtcorrelations -
                         semmvtcorrelations, meanmvtcorrelations + semmvtcorrelations, alpha=0.3)
axes1[0, 6].set_xlabel("Time (s)")
axes1[0, 6].set_ylabel("Mvt correlation")
axes1[0, 6].set_ylim([-0.1, 0.5])

axes1[0, 7].plot(timeWindow, meanturncorrelations)
axes1[0, 7].fill_between(timeWindow, meanturncorrelations -
                         semturncorrelations, meanturncorrelations + semturncorrelations, alpha=0.3)
axes1[0, 7].set_xlabel("Time (s)")
axes1[0, 7].set_ylabel("Ang vel correlation")
axes1[0, 7].set_ylim([-0.1, 0.5])


axes1[0, 0].plot(timeWindow, means["init"]["gcamp"], color="orange")
axes1[0, 0].fill_between(timeWindow, means["init"]["gcamp"] - sems["init"]["gcamp"],
                         means["init"]["gcamp"] + sems["init"]["gcamp"], color="orange", alpha=0.3)
axes1[0, 0].set_xlabel("Time (s)")
axes1[0, 0].set_ylabel("Zscore GCaMP")
axes1_1 = axes1[0, 0].twinx()
axes1_1.plot(timeWindow, means["init"]["move"], color="blue")
axes1_1.fill_between(timeWindow, means["init"]["move"] - sems["init"]["move"],
                     means["init"]["move"] + sems["init"]["move"], color="blue", alpha=0.3)
axes1_1.set_ylabel("Speed (cm/s)")
axes1[0, 0].set_ylim([-0.1, 0.4])
axes1_1.set_ylim([1, 7])

axes1[0, 1].plot(timeWindow, means["term"]["gcamp"], color="orange")
axes1[0, 1].fill_between(timeWindow, means["term"]["gcamp"] - sems["term"]["gcamp"],
                         means["term"]["gcamp"] + sems["term"]["gcamp"], color="orange", alpha=0.3)
axes1[0, 1].set_xlabel("Time (s)")
axes1[0, 1].set_ylabel("Zscore GCaMP")
axes1_1 = axes1[0, 1].twinx()
axes1_1.plot(timeWindow, means["term"]["move"], color="blue")
axes1_1.fill_between(timeWindow, means["term"]["move"] - sems["term"]["move"],
                     means["term"]["move"] + sems["term"]["move"], color="blue", alpha=0.3)
axes1_1.set_ylabel("Speed (cm/s)")
axes1[0, 1].set_ylim([-0.1, 0.4])
axes1_1.set_ylim([1, 7])

axes1[0, 2].plot(timeWindow, means["init"]["gcampleft"], color="orange")
axes1[0, 2].fill_between(timeWindow, means["init"]["gcampleft"] - sems["init"]["gcampleft"],
                         means["init"]["gcampleft"] + sems["init"]["gcampleft"], color="orange", alpha=0.3)
axes1[0, 2].set_xlabel("Time (s)")
axes1[0, 2].set_ylabel("Zscore GCaMP")
axes1_1 = axes1[0, 2].twinx()
axes1_1.plot(timeWindow, means["init"]["moveleft"], color="blue")
axes1_1.fill_between(timeWindow, means["init"]["moveleft"] - sems["init"]["moveleft"],
                     means["init"]["moveleft"] + sems["init"]["moveleft"], color="blue", alpha=0.3)
axes1[0, 2].set_ylim([-0.1, 0.4])
axes1_1.set_yticks(piyticks[3:-3])
axes1_1.set_yticklabels([f"{t/np.pi:.1g}π" if t != 0 else "0" for t in piyticks[3:-3]])
axes1_1.set_ylabel("Angular velocity (rad/s)")

axes1[0, 3].plot(timeWindow, means["term"]["gcampleft"], color="orange")
axes1[0, 3].fill_between(timeWindow, means["term"]["gcampleft"] - sems["term"]["gcampleft"],
                         means["term"]["gcampleft"] + sems["term"]["gcampleft"], color="orange", alpha=0.3)
axes1[0, 3].set_xlabel("Time (s)")
axes1[0, 3].set_ylabel("Zscore GCaMP")
axes1_1 = axes1[0, 3].twinx()
axes1_1.plot(timeWindow, means["term"]["moveleft"], color="blue")
axes1_1.fill_between(timeWindow, means["term"]["moveleft"] - sems["term"]["moveleft"],
                     means["term"]["moveleft"] + sems["term"]["moveleft"], color="blue", alpha=0.3)
axes1[0, 3].set_ylim([-0.1, 0.4])
axes1_1.set_yticks(piyticks[3:-3])
axes1_1.set_yticklabels([f"{t/np.pi:.1g}π" if t != 0 else "0" for t in piyticks[3:-3]])
axes1_1.set_ylabel("Angular velocity (rad/s)")

axes1[0, 4].plot(timeWindow, means["init"]["gcampright"], color="orange")
axes1[0, 4].fill_between(timeWindow, means["init"]["gcampright"] - sems["init"]["gcampright"],
                         means["init"]["gcampright"] + sems["init"]["gcampright"], color="orange", alpha=0.3)
axes1[0, 4].set_xlabel("Time (s)")
axes1[0, 4].set_ylabel("Zscore GCaMP")
axes1_1 = axes1[0, 4].twinx()
axes1_1.plot(timeWindow, means["init"]["moveright"], color="blue")
axes1_1.fill_between(timeWindow, means["init"]["moveright"] - sems["init"]["moveright"],
                     means["init"]["moveright"] + sems["init"]["moveright"], color="blue", alpha=0.3)
axes1[0, 4].set_ylim([-0.1, 0.4])
axes1_1.set_yticks(piyticks[3:-3])
axes1_1.set_yticklabels([f"{t/np.pi:.1g}π" if t != 0 else "0" for t in piyticks[3:-3]])
axes1_1.set_ylabel("Angular velocity (rad/s)")

axes1[0, 5].plot(timeWindow, means["term"]["gcampright"], color="orange")
axes1[0, 5].fill_between(timeWindow, means["term"]["gcampright"] - sems["term"]["gcampright"],
                         means["term"]["gcampright"] + sems["term"]["gcampright"], color="orange", alpha=0.3)
axes1[0, 5].set_xlabel("Time (s)")
axes1[0, 5].set_ylabel("Zscore GCaMP")
axes1_1 = axes1[0, 5].twinx()
axes1_1.plot(timeWindow, means["term"]["moveright"], color="blue")
axes1_1.fill_between(timeWindow, means["term"]["moveright"] - sems["term"]["moveright"],
                     means["term"]["moveright"] + sems["term"]["moveright"], color="blue", alpha=0.3)
axes1[0, 5].set_ylim([-0.1, 0.4])
axes1_1.set_yticks(piyticks[3:-3])
axes1_1.set_yticklabels([f"{t/np.pi:.1g}π" if t != 0 else "0" for t in piyticks[3:-3]])
axes1_1.set_ylabel("Angular velocity (rad/s)")

cax1 = axes1[1, 0].imshow(sorted(allallinitiationGCaMPSegments, key=lambda row: np.mean(
    row[:50])), cmap="magma", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(cax1, ax=axes1[1, 0])
cax1 = axes1[1, 1].imshow(sorted(allallterminationGCaMPSegments, key=lambda row: np.mean(
    row[:50])), cmap="magma", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(cax1, ax=axes1[1, 1])

cax1 = axes1[1, 2].imshow(sorted(allallinitiationleftGCaMPSegments, key=lambda row: np.mean(
    row[:50])), cmap="magma", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(cax1, ax=axes1[1, 2])
cax1 = axes1[1, 3].imshow(sorted(allallterminationleftGCaMPSegments, key=lambda row: np.mean(
    row[:50])), cmap="magma", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(cax1, ax=axes1[1, 3])

cax1 = axes1[1, 4].imshow(sorted(allallinitiationrightGCaMPSegments, key=lambda row: np.mean(
    row[:50])), cmap="magma", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(cax1, ax=axes1[1, 4])
cax1 = axes1[1, 5].imshow(sorted(allallterminationrightGCaMPSegments, key=lambda row: np.mean(
    row[:50])), cmap="magma", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(cax1, ax=axes1[1, 5])

plt.savefig(os.path.join(pathfiles, "Graph.png"), format="png")
plt.show()
print("All files have been processed")
