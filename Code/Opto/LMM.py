"""
Run LMM model on analyzed opto data
Author: Isaac Chang
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

def load():
    files = []
    for i in os.listdir(pathfiles):
        if os.path.splitext(i)[-1].lower() == ".csv":
            files.append(i)
    files.sort()
    return files

pathfiles = r"\\hive\Paz-Lab\Isaac\GPe\Data\Opto\LMM"

mani = pd.read_csv(os.path.join(pathfiles, "Npas1-eNpHR-sustained.csv"))
eYFP = pd.read_csv(os.path.join(pathfiles, "Npas1-eYFP-sustained.csv"))

maniPreSpeed = mani["Pre Speed"].dropna().values
maniSpeedChange = mani["Speed change"].dropna().values
maniPreDistance = mani["Pre Distance"].dropna().values
maniDistanceChange = mani["Distance change"].dropna().values

eYFPPreSpeed = eYFP["Pre Speed"].dropna().values
eYFPSpeedChange = eYFP["Speed change"].dropna().values
eYFPPreDistance = eYFP["Pre Distance"].dropna().values
eYFPDistanceChange = eYFP["Distance change"].dropna().values

preSpeed = np.concatenate((maniPreSpeed, eYFPPreSpeed))
speedChange = np.concatenate((maniSpeedChange, eYFPSpeedChange))
preDistance = np.concatenate((maniPreDistance, eYFPPreDistance))
distanceChange = np.concatenate((maniDistanceChange, eYFPDistanceChange))
condition = np.concatenate((
    np.full(len(maniPreSpeed), "Mani"),
    np.full(len(eYFPPreSpeed), "eYFP")
))
mouseID = np.repeat(np.arange(len(preSpeed) // 20), 20)

preSpeedLog = np.log10(preSpeed)
preSpeedLogC = np.log10(preSpeed) - np.log10(preSpeed).mean()
speedChangeLog = np.log10(speedChange)
preDistanceC = preDistance - preDistance.mean()

df = pd.DataFrame({
    "preSpeed": preSpeed,
    "preSpeedLog" : preSpeedLog,
    "preSpeedLogC": preSpeedLogC,
    "speedChange": speedChange,
    "speedChangeLog": speedChangeLog,
    "preDistance": preDistance,
    "preDistanceC": preDistanceC,
    "distanceChange": distanceChange,
    "condition": condition,
    "mouseID": mouseID
})

fig, axes = plt.subplots(
    2, 4, figsize=(20, 8)
)

palette = {
    "Mani": "blue",
    "eYFP": "black",
}

model = smf.mixedlm(
    "speedChangeLog ~ C(condition, Treatment(reference='eYFP')) * preSpeedLog",
    data=df,
    groups=df["mouseID"],
    re_formula="~preSpeedLog"
)

result = model.fit(method="lbfgs")

print(result.summary())

sns.scatterplot(
    data=df,
    x="preSpeedLog",
    y="speedChangeLog",
    alpha=0.4,
    hue="condition",
    palette = palette,
    ax = axes[0, 0]
)

sns.scatterplot(
    data=df,
    x="preSpeedLog",
    y="speedChangeLog",
    alpha=0.1,
    hue="condition",
    palette = palette,
    ax = axes[0, 1]
)

x1 = np.percentile(df["preSpeedLog"], 2.5)
x2 = np.percentile(df["preSpeedLog"], 97.5)
x = np.linspace(x1, x2, 100)

fe = result.fe_params
eYFPintercept = fe["Intercept"]
eYFPslope = fe["preSpeedLog"]
eYFPy = eYFPintercept + eYFPslope * x
eYFParea = np.sum(eYFPy)
axes[0, 1].plot(x , eYFPy, color = "black")
Maniintercept = eYFPintercept + fe["C(condition, Treatment(reference='eYFP'))[T.Mani]"]
Manislope = eYFPslope + fe["C(condition, Treatment(reference='eYFP'))[T.Mani]:preSpeedLog"]
Maniy = Maniintercept + Manislope * x
Maniarea = np.sum(Maniy)
axes[0, 1].plot(x , Maniy, color = "blue")

print(f"Speed lines")
print(f"eYFP: y = {eYFPslope:.2f}x + {eYFPintercept:.2f}, Area = {eYFParea:.2f}")
print(f"Mani: y = {Manislope:.2f}x + {Maniintercept:.2f}, Area = {Maniarea:.2f}")

slopes = []
intercepts = []
areas = []
posAreas = []
negAreas = []
conditions = []

for mouse in np.unique(df["mouseID"]):

    mousedata = df[df["mouseID"] == mouse]
    cond = mousedata["condition"].iloc[0]

    re = result.random_effects[mouse]

    if cond == "Mani":
        intercept = Maniintercept + re["Group"]
        slope = Manislope + re["preSpeedLog"]
        y = intercept + slope * x
        area = np.sum(y)
        posArea = np.sum(y[y>0])
        negArea = np.sum(y[y<0])
        axes[0, 0].plot(x , y, color = "blue", alpha = 0.5)
    elif cond == "eYFP":
        intercept = eYFPintercept + re["Group"]
        slope = eYFPslope + re["preSpeedLog"]
        y = intercept + slope * x
        area = np.sum(y)
        posArea = np.sum(y[y>0])
        negArea = np.sum(y[y<0])
        axes[0, 0].plot(x , y, color = "black", alpha = 0.5)
    
    slopes.append(slope)
    intercepts.append(intercept)
    areas.append(area)
    posAreas.append(posArea)
    negAreas.append(negArea)
    conditions.append(cond)

mousedf = pd.DataFrame({
    "slope": slopes,
    "area": areas,
    "posArea": posAreas,
    "negarea": negAreas,
    "intercept": intercepts,
    "condition": conditions
})

sns.stripplot(
    data=mousedf,
    x="condition",
    y="slope",
    palette = palette,
    ax=axes[0, 2]
)

sns.pointplot(
    data=mousedf,
    x="condition",
    y="slope",
    errorbar="se",
    join=False,
    palette =palette,
    markers="_",
    scale=2,
    errwidth=1.5,
    capsize=0.2,
    ax=axes[0, 2]
)

sns.stripplot(
    data=mousedf,
    x="condition",
    y="area",
    palette = palette,
    ax=axes[0,3]
)

sns.pointplot(
    data=mousedf,
    x="condition",
    y="area",
    errorbar="se",
    join=False,
    palette =palette,
    markers="_",
    scale=2,
    errwidth=1.5,
    capsize=0.2,
    ax=axes[0, 3]
)


axes[0, 1].fill_between(
    x,
    0,
    eYFPy,
    color="black",
    alpha=0.2
)

axes[0, 1].fill_between(
    x,
    0,
    Maniy,
    color="blue",
    alpha=0.2
)

axes[0, 0].set_xlabel("Log pre speed")
axes[0, 0].set_ylabel("Log fold change")
axes[0, 0].set_xlim(-1.5,1)
axes[0, 0].set_ylim(-2,2)
axes[0, 0].axhline(0, ls = ":")
axes[0, 1].set_xlabel("Log pre speed")
axes[0, 1].set_ylabel("Log fold change")
axes[0, 1].set_xlim(-1.5,1)
axes[0, 1].set_ylim(-1,1)
axes[0, 1].axhline(0, ls = ":")

outputfileName = "StatsSpeed.csv"
filePath = os.path.join(pathfiles, outputfileName)
mousedf.to_csv(filePath)

model = smf.mixedlm(
    "distanceChange ~ C(condition, Treatment(reference='eYFP')) * preDistance",
    data=df,
    groups=df["mouseID"],
    re_formula="~preDistance"
)

result = model.fit(method="lbfgs")

print(result.summary())

sns.scatterplot(
    data=df,
    x="preDistance",
    y="distanceChange",
    alpha=0.4,
    hue="condition",
    palette = palette,
    ax = axes[1, 0]
)

sns.scatterplot(
    data=df,
    x="preDistance",
    y="distanceChange",
    alpha=0.1,
    hue="condition",
    palette = palette,
    ax = axes[1, 1]
)

x1 = np.percentile(df["preDistance"], 2.5)
x2 = np.percentile(df["preDistance"], 97.5)
x = np.linspace(x1, x2, 100)

fe = result.fe_params
eYFPintercept = fe["Intercept"]
eYFPslope = fe["preDistance"]
eYFPy = eYFPintercept + eYFPslope * x
eYFParea = np.sum(eYFPy)
axes[1, 1].plot(x, eYFPy, color = "black")
Maniintercept = eYFPintercept + fe["C(condition, Treatment(reference='eYFP'))[T.Mani]"]
Manislope = eYFPslope + fe["C(condition, Treatment(reference='eYFP'))[T.Mani]:preDistance"]
Maniy = Maniintercept + Manislope * x
Maniarea = np.sum(Maniy)
axes[1, 1].plot(x, Maniy, color = "blue")

print(f"Distance lines")
print(f"eYFP: y = {eYFPslope:.2f}x + {eYFPintercept:.2f}, Area = {eYFParea:.2f}")
print(f"Mani: y = {Manislope:.2f}x + {Maniintercept:.2f}, Area = {Maniarea:.2f}")

slopes = []
intercepts = []
areas = []
posAreas = []
negAreas = []
conditions = []

for mouse in np.unique(df["mouseID"]):

    mousedata = df[df["mouseID"] == mouse]
    cond = mousedata["condition"].iloc[0]

    re = result.random_effects[mouse]

    if cond == "Mani":
        intercept = Maniintercept + re["Group"]
        slope = Manislope + re["preDistance"]
        y = intercept + slope * x
        area = np.sum(y)
        posArea = np.sum(y[y>0])
        negArea = np.sum(y[y<0])
        axes[1, 0].plot(x, y, color = "blue", alpha = 0.5)
    elif cond == "eYFP":
        intercept = eYFPintercept + re["Group"]
        slope = eYFPslope + re["preDistance"]
        y = intercept + slope * x
        area = np.sum(y)
        posArea = np.sum(y[y>0])
        negArea = np.sum(y[y<0])
        axes[1, 0].plot(x, y, color = "black", alpha = 0.5)
    
    slopes.append(slope)
    areas.append(area)
    posAreas.append(posArea)
    negAreas.append(negArea)
    intercepts.append(intercept)
    conditions.append(cond)

mousedf = pd.DataFrame({
    "slope": slopes,
    "area": areas,
    "posArea": posAreas,
    "negarea": negAreas,
    "intercept": intercepts,
    "condition": conditions
})

sns.stripplot(
    data=mousedf,
    x="condition",
    y="slope",
    palette = palette,
    ax=axes[1, 2]
)

sns.pointplot(
    data=mousedf,
    x="condition",
    y="slope",
    errorbar="se",
    join=False,
    palette =palette,
    markers="_",
    scale=2,
    errwidth=1.5,
    capsize=0.2,
    ax=axes[1, 2]
)


sns.stripplot(
    data=mousedf,
    x="condition",
    y="area",
    palette = palette,
    ax=axes[1, 3]
)

sns.pointplot(
    data=mousedf,
    x="condition",
    y="area",
    errorbar="se",
    join=False,
    palette =palette,
    markers="_",
    scale=2,
    errwidth=1.5,
    capsize=0.2,
    ax=axes[1, 3]
)


axes[1, 1].fill_between(
    x,
    0,
    eYFPy,
    color="black",
    alpha=0.2
)

axes[1, 1].fill_between(
    x,
    0,
    Maniy,
    color="blue",
    alpha=0.2
)

axes[1, 0].set_xlabel("Pre distance (cm)")
axes[1, 0].set_ylabel("Distance change (cm)")
axes[1, 0].set_xlim(0,15)
axes[1, 0].set_ylim(-8,8)
axes[1, 0].axhline(0, ls = ":")                      
axes[1, 1].set_xlabel("Pre distance (cm)")
axes[1, 1].set_ylabel("Distance change (cm)")
axes[1, 1].set_xlim(0,15)
axes[1, 1].set_ylim(-8,8)
axes[1, 1].axhline(0, ls = ":")   

outputfileName = "StatsDistance.csv"
filePath = os.path.join(pathfiles, outputfileName)
mousedf.to_csv(filePath)

fig.savefig(os.path.join(pathfiles, "Graph.png"), format="png")
