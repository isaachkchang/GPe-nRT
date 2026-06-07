"""
Generate an example video with angular velocity, speed and GCaMP signal for visualization

Author: Isaac Chang
"""

import cv2
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import numpy as np
import os
import pandas as pd
from scipy.signal import savgol_filter

pathfiles = r"\\hive\Paz-Lab\Isaac\Fiber\Analysis\Npas1_axons\combined\ic240713i-240920-120952_processed.csv"
pathvideo = r"P:\Isaac\DLC\videos\fiber\FiberphotometryTest-240920_ic240713i-240920-120952_Vid1DLC_Resnet50_GPEDec9shuffle3_snapshot_040_filtered_p0_labeled.mp4"
pathvideo2 = r"C:\Users\isaac.chang\My Drive\Code\tdt\output.mp4"

START = 5600
END = 6200
N = END - START
# Speed threshold for movement onset/offset
onoffsetThreshold = 2
# Higher speed threshold to qualify as real movement
mvtThreshold = 5
# Angular velocity threshold for left/right turns. Left is negative. Right is positive
angVelThreshold = np.pi/2
cmap = "magma"

df = pd.read_csv(pathfiles)
GCaMP = df["GCaMP"].to_numpy()
speed = df["Speed"].to_numpy()
speed = np.convolve(speed, np.ones(15)/15, mode="same")
heading = df["Heading"].to_numpy()
angVel = df["Angular Velocity"].to_numpy()
angVel = savgol_filter(angVel, 5, 3)
# Not sure why, but left and right is flipped visually
angVel = -angVel
piyticks = np.arange(-4, 4.5, 1) * np.pi
GCaMP = GCaMP[START-100:END-100]
speed = speed[START-100:END-100]
angVel = angVel[START-100:END-100]
heading = heading[START-100:END-100]

# Detect movement bouts based on thresholds
mvtcutoff = speed > onoffsetThreshold
onsetmvt = np.where((mvtcutoff[1:] & ~mvtcutoff[:-1]))[0] + 1
offsetmvt = np.where((~mvtcutoff[1:] & mvtcutoff[:-1]))[0] + 1
if onsetmvt.size != offsetmvt.size:
    offsetmvt = offsetmvt[1:]  # Ensure matched onsets and offsets
onsetbout = []
offsetbout = []
for index2, value2 in enumerate(onsetmvt):
    if index2 < offsetmvt.size:
        if np.max(speed[value2: offsetmvt[index2]]) > mvtThreshold:
            onsetbout.append(onsetmvt[index2])
            offsetbout.append(offsetmvt[index2])

# Detect left/rught turns based on thresholds
leftmask = angVel < -angVelThreshold
rightmask = angVel > angVelThreshold
onsetleft = np.where(np.diff(leftmask.astype(int)) == 1)[0] + 1
offsetleft = np.where(np.diff(leftmask.astype(int)) == -1)[0] + 1
onsetright = np.where(np.diff(rightmask.astype(int)) == 1)[0] + 1
offsetright = np.where(np.diff(rightmask.astype(int)) == -1)[0] + 1

cap = cv2.VideoCapture(pathvideo)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, START)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fig = plt.figure(figsize=(50, 10))

ax_video = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
ax_angVel = plt.subplot2grid((3, 2), (0, 1))
ax_mvt = plt.subplot2grid((3, 2), (1, 1))
ax_gcamp = plt.subplot2grid((3, 2), (2, 1))


ax_angVel.plot(angVel, color="purple")
scroll_d = ax_angVel.axvline(0, color="red")
for onset, offset in zip(onsetleft, offsetleft):
    ax_angVel.axvspan(onset, offset, color='orange', alpha=0.3)
for onset, offset in zip(onsetright, offsetright):
    ax_angVel.axvspan(onset, offset, color='purple', alpha=0.3)
ax_angVel.set_ylabel("Angular velocity (rad/s)")
ax_angVel.set_xlim(0, N)
ax_angVel.set_ylim(min(angVel), max(angVel))
ax_angVel.set_yticks(piyticks[2:-2])
ax_angVel.set_yticklabels(
    [f"{t/np.pi:.1g}π" if t != 0 else "0" for t in piyticks[2:-2]])


ax_mvt.plot(speed, color="blue")
scroll_m = ax_mvt.axvline(0, color="red")
for onset, offset in zip(onsetbout, offsetbout):
    ax_mvt.axvspan(onset, offset, color='blue', alpha=0.3)
ax_mvt.set_ylabel("Speed (cm/s)")
ax_mvt.set_xlim(0, N)
ax_mvt.set_ylim(0, 12)

ax_gcamp.plot(GCaMP, color="red")
scroll_g = ax_gcamp.axvline(0, color="red")
ax_gcamp.set_ylabel("Z-score GCaMP")
ax_gcamp.set_xlim(0, N)
ax_gcamp.set_ylim(-2, 3)

img_display = ax_video.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
ax_video.axis("off")


def update(i):
    frame_i = START + i
    ret, frame = cap.read()
    if not ret or frame_i >= END:
        return (img_display, scroll_m, scroll_g)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img_display.set_data(frame)

    scroll_d.set_xdata([i])
    scroll_m.set_xdata([i])
    scroll_g.set_xdata([i])

    return (img_display,
            scroll_d,
            scroll_m,
            scroll_g)

cap.set(cv2.CAP_PROP_POS_FRAMES, START)

ret, frame = cap.read()

anim = FuncAnimation(
    fig,
    update,
    frames=N,
    interval=1000/fps,
    blit=True
)

plt.show()

cap.set(cv2.CAP_PROP_POS_FRAMES, START)
writer = FFMpegWriter(fps=fps, bitrate=5000)
anim.save("output.mp4", writer=writer, dpi=200,
          progress_callback=lambda i, n: print(f"{i}/{n} frames"))
