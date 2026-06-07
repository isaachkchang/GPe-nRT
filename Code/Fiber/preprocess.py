"""
Preprocess TDT fiber photometry recordings alongside DLC movement tracking data. 

Author: Isaac Chang
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import tdt

def loadFiber():
    files = []
    for i in os.listdir(pathfiles):
        if os.path.isdir(os.path.join(pathfiles, i)):
            files.append(i)
    files.sort()
    return files

def loadMvt():
    files = []
    for i in os.listdir(pathfiles):
        if os.path.splitext(i)[-1].lower() == ".csv":
            files.append(i)
    files.sort()
    return files

def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    tau_fast = tau_slow*tau_multiplier
    y = const+amp_slow*np.exp(-t/tau_slow)+amp_fast*np.exp(-t/tau_fast)
    return y


pathfiles = r"\\hive\Paz-Lab\Isaac\GPe\Data\Fiber\Preprocessing\Fiber"
fiberFiles = loadFiber()
mvtFiles = loadMvt()

ISOS = '_405A'
GCaMP = '_465A'
Vid = 'Vid1'
timeCutoff = 10

print("Analyzing " + str(len(fiberFiles)) + " fiber photometry files below:")
print(fiberFiles)
print("Analyzing " + str(len(mvtFiles)) + " movement files below:")
print(mvtFiles)

for fiber, mvt in zip(fiberFiles, mvtFiles):

    data = tdt.read_block(os.path.join(pathfiles, fiber))
    samplingRate = data.streams[ISOS].fs
    rawISOS = data.streams[ISOS].data[int(timeCutoff * data.streams[ISOS].fs):]
    rawGCaMP = data.streams[GCaMP].data[int(
        timeCutoff * data.streams[GCaMP].fs):]

    if len(rawISOS) <= len(rawGCaMP):
        rawGCaMP = rawGCaMP[:len(rawISOS)]
    else:
        rawISOS = rawISOS[:len(rawGCaMP)]

    timeISOS = np.linspace(1, len(rawISOS), len(rawISOS)
                           ) / data.streams[ISOS].fs
    timeGCaMP = np.linspace(1, len(rawGCaMP), len(
        rawGCaMP)) / data.streams[GCaMP].fs
    timeVid = data.epocs[Vid].offset

    df = pd.read_csv(os.path.join(pathfiles, mvt))
    speed = df["speed"].to_numpy()
    heading = df["heading"].to_numpy()
    angVel = df["angular_velocity"].to_numpy()
    speed = speed[np.where(timeVid <= timeCutoff)[0][-1]:]
    heading = heading[np.where(timeVid <= timeCutoff)[0][-1]:]
    angVel = angVel[np.where(timeVid <= timeCutoff)[0][-1]:]
    timeVid = timeVid[timeVid >= timeCutoff] - timeCutoff
    

    fig, axes = plt.subplots(5, 2, figsize=(30, 10))
    
    axes[0, 0].plot(timeISOS, rawISOS, color='cornflowerblue')
    axes[0, 0].set_ylim(np.mean(rawISOS)-5*np.std(rawISOS),
                        np.mean(rawISOS)+5*np.std(rawISOS))
    axes[0, 1].plot(timeGCaMP, rawGCaMP, color='firebrick')
    axes[0, 1].set_ylim(np.mean(rawGCaMP)-3*np.std(rawGCaMP),
                        np.mean(rawGCaMP)+10*np.std(rawGCaMP))
    
    axes[0, 0].set_ylabel("Raw ISOS signal")
    axes[0, 1].set_ylabel("Raw GCaMP signal")

    b, a = butter(2, 2, btype='low', fs=samplingRate)
    filteredISOS = filtfilt(b, a, rawISOS)
    filteredGCaMP = filtfilt(b, a, rawGCaMP)
    max_sig = np.max(filteredISOS)
    inital_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
    bounds = ([0, 0, 0, 600, 0],
              [max_sig, max_sig, max_sig, 36000, 1])
    paramsISOS, cv = curve_fit(double_exponential, timeISOS, filteredISOS,
                               p0=inital_params, bounds=bounds, maxfev=1000)
    expfitISOS = double_exponential(timeISOS, *paramsISOS)
    max_sig = np.max(filteredGCaMP)
    inital_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
    bounds = ([0, 0, 0, 600, 0],
              [max_sig, max_sig, max_sig, 36000, 1])
    paramsGCaMP, cv = curve_fit(double_exponential, timeGCaMP, filteredGCaMP,
                                p0=inital_params, bounds=bounds, maxfev=1000)
    expfitGCaMP = double_exponential(timeGCaMP, *paramsGCaMP)

    axes[1, 0].plot(timeISOS, filteredISOS, color='cornflowerblue')
    axes[1, 0].plot(timeISOS, expfitISOS, color='black')
    axes[1, 0].plot(timeISOS, rawISOS, color='cornflowerblue', alpha=0.5)
    axes[1, 0].set_ylim(np.mean(filteredISOS)-5*np.std(filteredISOS),
                        np.mean(filteredISOS)+5*np.std(filteredISOS))
    

    axes[1, 1].plot(timeGCaMP, filteredGCaMP, color='firebrick')
    axes[1, 1].plot(timeGCaMP, expfitGCaMP, color='black')
    axes[1, 1].plot(timeGCaMP, rawGCaMP, color='firebrick', alpha=0.5)
    axes[1, 1].set_ylim(np.mean(filteredGCaMP)-3*np.std(filteredGCaMP),
                        np.mean(filteredGCaMP)+10*np.std(filteredGCaMP))
    
    axes[1, 0].set_ylabel("Filtered ISOS signal")
    axes[1, 1].set_ylabel("Filtered GCaMP signal")

    detrendedISOS = filteredISOS - expfitISOS
    detrendedGCaMP = filteredGCaMP - expfitGCaMP

    axes[2, 0].plot(timeISOS, detrendedISOS, color='cornflowerblue')
    axes[2, 0].plot(timeISOS, filteredISOS, color='cornflowerblue', alpha=0.5)
    axes[2, 0].set_ylim(np.mean(detrendedISOS)-5*np.std(detrendedISOS),
                        np.mean(detrendedISOS)+5*np.std(detrendedISOS))
    axes[2, 1].plot(timeGCaMP, detrendedGCaMP, color='firebrick')
    axes[2, 1].plot(timeGCaMP, filteredGCaMP, color='firebrick', alpha=0.5)
    axes[2, 1].set_ylim(np.mean(detrendedGCaMP) - 5*np.std(detrendedGCaMP),
                        np.mean(detrendedGCaMP) + 5*np.std(detrendedGCaMP))
    
    axes[2, 0].set_ylabel("Detrended ISOS signal")
    axes[2, 1].set_ylabel("Detrended GCaMP signal")


    slope, intercept, r_value, p_value, std_err = linregress(
        x=detrendedISOS, y=detrendedGCaMP)
    print("Slope = " + str(slope))
    print("intercept = " + str(intercept))
    print("R value  = " + str(r_value))

    estMotionGCaMP = slope * detrendedISOS
    correctedGCaMP = detrendedGCaMP - estMotionGCaMP

    axes[3, 0].plot(timeGCaMP, correctedGCaMP, color='firebrick')
    axes[3, 0].plot(timeGCaMP, estMotionGCaMP, color='orange')
    axes[3, 0].plot(timeGCaMP, detrendedGCaMP, color='firebrick', alpha=0.5)
    axes[3, 0].set_ylim(np.mean(correctedGCaMP)-5*np.std(correctedGCaMP),
                        np.mean(correctedGCaMP)+5*np.std(correctedGCaMP))
    
    axes[3, 0].set_ylabel("Corrected GCaMP signal")

    zscoreGCaMP = (correctedGCaMP-np.mean(correctedGCaMP)) / \
        np.std(correctedGCaMP)
    axes[3, 1].plot(timeGCaMP, zscoreGCaMP, color='firebrick')
    axes[3, 1].set_ylim(np.mean(zscoreGCaMP)-5*np.std(zscoreGCaMP),
                        np.mean(zscoreGCaMP)+5*np.std(zscoreGCaMP))

    axes[3, 1].set_ylabel("Z-score GCaMP signal")

    resampleGCaMP = np.zeros(len(timeVid))
    for i in range(len(timeVid)-1):
        startTime = timeVid[i]
        endTime = timeVid[i + 1]
        mask = (timeGCaMP >= startTime) & (timeGCaMP < endTime)
        if np.any(mask):
            resampleGCaMP[i] = np.mean(zscoreGCaMP[mask])
        else:
            resampleGCaMP[i] = np.nan
    mask = (timeGCaMP >= timeVid[-1])
    if np.any(mask):
        resampleGCaMP[-1] = np.mean(resampleGCaMP[mask])
    else:
        resampleGCaMP[-1] = np.nan

    axes[4, 0].plot(timeGCaMP, correctedGCaMP, color='firebrick', alpha=0.5)
    axes[4, 0].plot(timeVid, resampleGCaMP, color='firebrick')
    
    axes[4, 0].set_ylabel("Resampled GCaMP signal")
    axes[4, 0].set_xlabel("Time (s)")

    axes[4, 1].plot(timeVid, resampleGCaMP, color='firebrick')

    axes[4, 1].set_ylabel("Zscore GCaMP")
    axes[4, 1].set_xlabel("Time (s)")
    axes_1 = axes[4, 1].twinx()
    axes_1.set_ylabel("Speed (cm/s)")
    axes_1.plot(timeVid, speed[:len(timeVid)], color='blue')

    plt.savefig(os.path.join(pathfiles, fiber + " _Graph.png"), format="png")
    plt.close()

    outdf = pd.DataFrame()

    outdf["GCaMP"] = resampleGCaMP
    outdf["Speed"] = speed[:len(timeVid)]
    outdf["Heading"] = heading[:len(timeVid)]
    outdf["Angular Velocity"] = angVel[:len(timeVid)]
    filePath = os.path.join(pathfiles, fiber + "_processed.csv")
    outdf.to_csv(filePath)
    print(fiber + " has been processed")
    print(mvt + " has been processed")

print("All files have been processed")
