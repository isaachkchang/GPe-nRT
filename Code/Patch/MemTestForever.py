"""
Analzye response to -10mV test pulse, for many sweeps.
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


def step_calculate(abf, trace):
    assert isinstance(abf, pyabf.ABF)

    stepTrace = trace[stepStart:stepEnd]
    Istep = -np.mean(stepTrace[int(len(stepTrace) * 0.5):])
    Ipeak = np.min(stepTrace)
    Ra = abs(((dV * (1e-3)) / (Ipeak * (1e-12))) * (1e-6))
    Rs = abs(((dV * (1e-3)) / (Istep * (1e-12))) * (1e-6))
    Rm = abs(dV) * (1e-3) / (Istep * (1e-12)) * (1e-6) - Ra

    return Rm, Ra, Rs


path_in = r"C:\Users\isaac.chang\Desktop\Data\Patch\KCl_Opto\MemTestForever"
path_out = path_in

files = load(path_in)
print("Analyzing " + str(len(files)) + " files below:")
print(files)

df = pd.DataFrame()

for index, value in enumerate(files):
    abf = pyabf.ABF(os.path.join(path_in, value))

    dV = -10
    stepStart = int(106.3 * abf.dataPointsPerMs)
    stepEnd = int(206.3 * abf.dataPointsPerMs)
    Ra = np.zeros(abf.sweepCount)
    Rs = np.zeros(abf.sweepCount)
    Rm = np.zeros(abf.sweepCount)

    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber, baseline=[0, 0.1])
        Rm[sweepNumber], Ra[sweepNumber], Rs[sweepNumber] = step_calculate(
            abf, abf.sweepY)

    df = pd.concat([df, pd.Series(dtype="float64", name=index).to_frame().T])
    df.loc[index, "File name"] = str(os.path.basename(abf.abfFilePath))
    df.loc[index, "Initial access resistance"] = Ra[0]
    df.loc[index, "Initial seal resistance"] = Rs[0]
    df.loc[index, "Final access resistance"] = Ra[-1]
    df.loc[index, "Final membrane resistance"] = Rm[-1]

    for index2, value2 in enumerate(Ra):
        df.loc[index, str(index2)] = value2

    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")

fileName = "MemTestForever_Results.csv"
filePath = os.path.join(path_out, fileName)
df.to_csv(filePath)
print("All files have been processed")
