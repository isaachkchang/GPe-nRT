"""
Analzye CSV files containing Z-scored GCaMP signals and movement data, for individual mouse.
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
cmap = "magma"

print("Analyzing " + str(len(files)) + " files below:")
print(files)

outdf = pd.DataFrame()

for index, value in enumerate(files):

    df = pd.read_csv(os.path.join(pathfiles, value))
    GCaMP = df["GCaMP"].to_numpy()
    speed = df["Speed"].to_numpy()
    speed = np.convolve(speed, np.ones(15)/15, mode="same")
    heading = df["Heading"].to_numpy()
    angVel = df["Angular Velocity"].to_numpy()
    angVel = savgol_filter(angVel, 5, 3)
    angVel = -angVel


    fig1, axes1 = plt.subplots(
        4, 4, figsize=(20, 10), gridspec_kw={"width_ratios": [5, 2, 2, 2]}
    )

    axes1[0, 0].plot(GCaMP, color="firebrick", alpha=0.5)
    axes1[0, 0].set_ylabel("Zscore GCaMP")
    axes1[0, 0].set_xlabel("Time (s)")
    axes1_1 = axes1[0, 0].twinx()
    axes1_1.set_ylabel("Speed (cm/s)")
    axes1_1.plot(speed, color="blue", alpha=0.5)

    piyticks = np.arange(-4, 4.5, 1) * np.pi
    axes1[2, 0].plot(GCaMP, color="firebrick", alpha=0.5)
    axes1[2, 0].set_ylabel("Zscore GCaMP")
    axes1[2, 0].set_xlabel("Time (s)")
    axes1_3 = axes1[2, 0].twinx()
    axes1_3.set_yticks(piyticks)
    axes1_3.set_yticklabels(
        [f"{t/np.pi:.1g}π" if t != 0 else "0" for t in piyticks])
    axes1_3.set_ylabel("Angular velocity (rad/s)")
    axes1_3.plot(angVel, color="blue", alpha=0.5)

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

    axes1[1, 0].plot(GCaMP, color="firebrick", alpha=0.5)
    axes1[1, 0].set_ylabel("Zscore GCaMP")
    axes1[1, 0].set_xlabel("Time (s)")
    axes1_2 = axes1[1, 0].twinx()
    axes1_2.set_ylabel("Speed (cm/s)")
    axes1_2.plot(speed, color="blue", alpha=0.5)
    axes1[1, 0].set_xlim(26000, 32000)
    for onset, offset in zip(onsetbout, offsetbout):
        axes1_2.axvspan(onset, offset, color='blue', alpha=0.3)
    

    axes1[3, 0].plot(GCaMP, color="firebrick", alpha=0.5)
    axes1[3, 0].set_ylabel("Zscore GCaMP")
    axes1[3, 0].set_xlabel("Time (s)")
    axes1_4 = axes1[3, 0].twinx()
    axes1_4.set_yticks(piyticks[2:-2])
    axes1_4.set_yticklabels(
        [f"{t/np.pi:.1g}π" if t != 0 else "0" for t in piyticks[2:-2]])
    axes1_4.set_ylabel("Angular Velocity (radian/s)")
    axes1_4.plot(angVel, color="blue", alpha=0.5)
    axes1_4.set_ylim(-np.pi*2, np.pi*2)
    axes1[3, 0].set_xlim(5600, 6200)
    for onset, offset in zip(onsetleft, offsetleft):
        axes1_4.axvspan(onset, offset, color='orange', alpha=0.3)
    for onset, offset in zip(onsetright, offsetright):
        axes1_4.axvspan(onset, offset, color='purple', alpha=0.3)

    mvtcorrelations = [correlation(speed, GCaMP, lag) for lag in cortimeWindow]
    axes1[3, 1].plot(timeWindow, mvtcorrelations)
    axes1[3, 1].set_ylabel("Mvt correlation")
    axes1[3, 1].set_xlabel("Time (s)")
    turncorrelations = [correlation(angVel, GCaMP, lag) for lag in cortimeWindow]
    axes1[3, 2].plot(timeWindow, turncorrelations)
    axes1[3, 2].set_ylabel("Ang vel coefficient")
    axes1[3, 2].set_xlabel("Time (s)")
    

    initiationMvtSegments = []
    initiationGCaMPSegments = []
    terminationMvtSegments = []
    terminationGCaMPSegments = []
    initiationleftMvtSegments = []
    initiationleftGCaMPSegments = []
    terminationleftMvtSegments = []
    terminationleftGCaMPSegments = []
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
            else:
                break
    for n in offsetbout:
        if n > bouttimeWindow:
            start = int(n - bouttimeWindow)
            end = start + 100
            if end < len(speed):
                terminationMvtSegments.append(speed[start:end])
                terminationGCaMPSegments.append(GCaMP[start:end])
            else:
                break
    for n in onsetleft:
        if n > bouttimeWindow:
            start = int(n - bouttimeWindow)
            end = start + 100
            if end < len(angVel):
                initiationleftMvtSegments.append(angVel[start:end])
                initiationleftGCaMPSegments.append(GCaMP[start:end])
            else:
                break
    for n in offsetleft:
        if n > bouttimeWindow:
            start = int(n - bouttimeWindow)
            end = start + 100
            if end < len(angVel):
                terminationleftMvtSegments.append(angVel[start:end])
                terminationleftGCaMPSegments.append(GCaMP[start:end])
            else:
                break
    for n in onsetright:
        if n > bouttimeWindow:
            start = int(n - bouttimeWindow)
            end = start + 100
            if end < len(angVel):
                initiationrightMvtSegments.append(angVel[start:end])
                initiationrightGCaMPSegments.append(GCaMP[start:end])
            else:
                break
    for n in offsetright:
        if n > bouttimeWindow:
            start = int(n - bouttimeWindow)
            end = start + 100
            if end < len(angVel):
                terminationrightMvtSegments.append(angVel[start:end])
                terminationrightGCaMPSegments.append(GCaMP[start:end])
            else:
                break

    axes1[0, 1].plot(
        timeWindow, np.mean(initiationGCaMPSegments, axis=0), color="orange"
    )
    axes1[0, 1].set_xlabel("Time (s)")
    axes1[0, 1].set_ylabel("Zscore GCaMP")
    axes1_2 = axes1[0, 1].twinx()
    axes1_2.plot(timeWindow, np.mean(
        initiationMvtSegments, axis=0), color="blue")
    axes1_2.set_ylabel("Speed (cm/s)")

    axes1[0, 2].plot(
        timeWindow, np.mean(initiationleftGCaMPSegments, axis=0), color="orange"
    )
    axes1[0, 2].set_xlabel("Time (s)")
    axes1[0, 2].set_ylabel("Zscore GCaMP")
    axes1_2 = axes1[0, 2].twinx()
    axes1_2.plot(timeWindow, np.mean(
        initiationleftMvtSegments, axis=0), color="blue")
    axes1_2.set_ylabel("Angular velocity (rad/s)")

    axes1[0, 3].plot(
        timeWindow, np.mean(initiationrightGCaMPSegments, axis=0), color="orange"
    )
    axes1[0, 3].set_xlabel("Time (s)")
    axes1[0, 3].set_ylabel("Zscore GCaMP")
    axes1_2 = axes1[0, 3].twinx()
    axes1_2.plot(timeWindow, np.mean(
        initiationrightMvtSegments, axis=0), color="blue")
    axes1_2.set_ylabel("Angular velocity (rad/s)")

    axes1[1, 1].plot(
        timeWindow, np.mean(terminationGCaMPSegments, axis=0), color="orange"
    )
    axes1[1, 1].set_xlabel("Time (s)")
    axes1[1, 1].set_ylabel("Zscore GCaMP")
    axes1_2 = axes1[1, 1].twinx()
    axes1_2.plot(timeWindow, np.mean(
        terminationMvtSegments, axis=0), color="blue")
    axes1_2.set_ylabel("Speed (cm/s)")

    axes1[1, 2].plot(
        timeWindow, np.mean(terminationleftGCaMPSegments, axis=0), color="orange"
    )
    axes1[1, 2].set_xlabel("Time (s)")
    axes1[1, 2].set_ylabel("Zscore GCaMP")
    axes1_2 = axes1[1, 2].twinx()
    axes1_2.plot(timeWindow, np.mean(
        terminationleftMvtSegments, axis=0), color="blue")
    axes1_2.set_ylabel("Angular velocity (rad/s)")

    axes1[1, 3].plot(
        timeWindow, np.mean(terminationrightGCaMPSegments, axis=0), color="orange"
    )
    axes1[1, 3].set_xlabel("Time (s)")
    axes1[1, 3].set_ylabel("Zscore GCaMP")
    axes1_2 = axes1[1, 3].twinx()
    axes1_2.plot(timeWindow, np.mean(
        terminationrightMvtSegments, axis=0), color="blue")
    axes1_2.set_ylabel("Angular velocity (rad/s)")

    axes1[2, 1].scatter(speed, GCaMP, s=0.1)
    axes1[2, 1].set_ylabel("Zscore GCaMP")
    axes1[2, 1].set_xlabel("Speed (cm/s)")
    axes1[2, 1].set_ylim(-4, 5)
    axes1[2, 2].scatter(angVel, GCaMP, s=0.1)
    axes1[2, 2].set_ylabel("Zscore GCaMP")
    axes1[2, 2].set_xlabel("Angular velocity (rad/s)")
    axes1[2, 2].set_xlim(-np.pi*2, np.pi*2)
    axes1[2, 2].set_ylim(-4, 5)
    axes1[2, 2].set_xticks(piyticks[2:-2])
    axes1[2, 2].set_xticklabels([f"{t/np.pi:.1g}π" if t != 0 else "0" for t in piyticks[2:-2]])
    
    plt.savefig(os.path.join(pathfiles, value + " _Graph.svg"), format="svg")
    plt.close()

    print(value + " has been processed")
