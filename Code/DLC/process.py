"""
Extract movement kinematics from filtered DLC files. Second step of preprocessing
Author: Isaac Chang
"""

import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os
import pandas as pd


def load(path_in):
    files = []
    for i in os.listdir(path_in):
        if i.lower().endswith("filtered.h5"):
            files.append(i)
    files.sort()
    return files


def update(i):

    frame = START_FRAME + i

    if frame >= len(cx):
        return (nose_dot, *artists.values(), speed_line, angvel_line)
    
    nose_dot.set_data(
        nose_x_rot[frame],
        nose_y_rot[frame]
    )

    for name, (x, y, _) in bp.items():
        artists[name].set_data(x[frame], y[frame])

    start = max(0, frame - WIN_FRAMES)
    end = frame + 1
    if end - start < 2:
        return (nose_dot, *artists.values(), speed_line, angvel_line)

    t = np.arange(start, end) / FPS

    speed_line.set_data(t, speed[start:end])
    angvel_line.set_data(t, ang_vel[start:end])

    ax_speed.set_xlim(t[0], t[-1])
    ax_angvel.set_xlim(t[0], t[-1])

    print(str(frame))

    return (
        nose_dot,
        *artists.values(),
        speed_line,
        angvel_line
    )


path_in = r"\\hive\Paz-Lab\Isaac\DLC\videos\opto\Npas1-eNpHR-sustained"
path_out = path_in

files = load(path_in)

print("Analyzing " + str(len(files)) + " files below:")
print(files)

START_FRAME = 3000
N_FRAMES = 360
FPS = 10
WIN_SEC = 10
WIN_FRAMES = int(WIN_SEC * FPS)
PIXEL_TO_CM = 400/30
SPEED_THRESHOLD = 20

for index, value in enumerate(files):
    df = pd.read_hdf(os.path.join(path_in, value), key="df",
                     header=[0, 1, 2], index_col=0)

    scorer = df.columns.levels[0][0]
    nx = df[scorer]["Nose"]["x"].to_numpy()
    ny = df[scorer]["Nose"]["y"].to_numpy()
    lex = df[scorer]["Left_ear"]["x"].to_numpy()
    ley = df[scorer]["Left_ear"]["y"].to_numpy()
    rex = df[scorer]["Right_ear"]["x"].to_numpy()
    rey = df[scorer]["Right_ear"]["y"].to_numpy()
    cx = df[scorer]["Centroid"]["x"].to_numpy()
    cy = df[scorer]["Centroid"]["y"].to_numpy()
    lhx = df[scorer]["Left_lateral"]["x"].to_numpy()
    lhy = df[scorer]["Left_lateral"]["y"].to_numpy()
    rhx = df[scorer]["Right_lateral"]["x"].to_numpy()
    rhy = df[scorer]["Right_lateral"]["y"].to_numpy()
    tx = df[scorer]["Tail_base"]["x"].to_numpy()
    ty = df[scorer]["Tail_base"]["y"].to_numpy()

    bp = {
        "Nose":      (nx,  ny,  "purple"),
        "Centroid":  (cx,  cy,  "black"),
        "Tail":      (tx,  ty,  "blue"),
        "L_ear":     (lex, ley, "green"),
        "R_ear":     (rex, rey, "green"),
        "L_hip":     (lhx, lhy, "orange"),
        "R_hip":     (rhx, rhy, "orange"),
    }

    dcx = np.diff(cx)
    dcy = np.diff(cy)
    speed = np.concatenate(
        [[np.nan], np.sqrt(dcx**2 + dcy**2) * FPS]) / PIXEL_TO_CM
    speed[speed > SPEED_THRESHOLD] = SPEED_THRESHOLD

    v1x = nx - cx
    v1y = ny - cy
    v2x = tx - cx
    v2y = ty - cy
    dot = v1x * v2x + v1y * v2y
    norm1 = np.sqrt(v1x**2 + v1y**2)
    norm2 = np.sqrt(v2x**2 + v2y**2)
    cos_theta = dot / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    cross = v1x * v2y - v1y * v2x
    signed_angle_rad = np.sign(cross) * np.arccos(cos_theta)
    signed_angle_unwrapped = np.unwrap(signed_angle_rad)

    ang_vel = np.concatenate([[np.nan], np.diff(signed_angle_unwrapped) * FPS])

    tail_angle = np.arctan2(v2y, v2x)
    rot_angle = -tail_angle - np.pi / 2

    cos_r = np.cos(rot_angle)
    sin_r = np.sin(rot_angle)
    nose_x_rot = (
        v1x * cos_r - v1y * sin_r
    )
    nose_y_rot = (
        v1x * sin_r + v1y * cos_r
    )

    fig = plt.figure(figsize=(14, 8))

    ax_body = fig.add_subplot(2, 2, 1)
    ax_arena = fig.add_subplot(2, 2, 2)
    ax_speed = fig.add_subplot(2, 2, 3)
    ax_angvel = fig.add_subplot(2, 2, 4)

    ax_body.scatter(0, 0, c="red", s=50)    
    ax_body.arrow(0, 0, 0, -50, head_width=5, color="blue")
    ax_body.axhline(0, color="k", lw=1)
    ax_body.axvline(0, color="k", lw=1)

    ax_body.set_xlim(-150, 150)
    ax_body.set_ylim(-150, 150)
    ax_body.set_aspect("equal")
    ax_body.set_title("Body-centered frame")
    ax_body.set_xlabel("Left ⟷ Right")
    ax_body.set_ylabel("Forward ⟷ Backward")

    nose_dot, = ax_body.plot([], [], "o", color="purple", markersize=6)

    ax_arena.set_aspect("equal")
    ax_arena.set_title("Arena coordinates")
    ax_arena.set_xlabel("X (px)")
    ax_arena.set_ylabel("Y (px)")
    ax_arena.invert_yaxis()

    ax_arena.set_xlim(np.nanmin(cx)-50, np.nanmax(cx)+50)
    ax_arena.set_ylim(np.nanmin(cy)-50, np.nanmax(cy)+50)

    t_speed = np.arange(len(speed)) / FPS

    ax_speed.set_title("Speed (scrolling)")
    ax_speed.set_ylabel("Speed (px/s)")
    ax_speed.set_xlabel("Time (s)")

    speed_line, = ax_speed.plot([], [], color="black")
    ax_speed.set_ylim(0, np.nanpercentile(speed, 99))

    t_ang = np.arange(len(ang_vel)) / FPS

    ax_angvel.set_title("Angular velocity (scrolling)")
    ax_angvel.set_ylabel("deg/s")
    ax_angvel.set_xlabel("Time (s)")

    angvel_line, = ax_angvel.plot([], [], color="purple")
    ax_angvel.set_ylim(
        -np.nanpercentile(np.abs(ang_vel), 99),
        np.nanpercentile(np.abs(ang_vel), 99)
    )

    artists = {}
    for name, (_, _, color) in bp.items():
        artists[name], = ax_arena.plot([], [], "o", color=color, label=name)

    ani = FuncAnimation(
        fig,
        update,
        frames=N_FRAMES,
        interval=int(300 / FPS),
        blit=False
    )
    #writer = PillowWriter(fps=FPS*6)
    #ani.save("test.gif", writer=writer)
    #plt.show()
    outdf = pd.DataFrame({
        "frame": np.arange(len(speed)),
        "speed": speed,
        "heading": signed_angle_unwrapped,
        "angular_velocity": ang_vel,
        "XCor": cx/PIXEL_TO_CM,
        "YCor": cy/PIXEL_TO_CM
    })

    outname = value.replace(
        "_Vid1DLC_Resnet50_GPEDec9shuffle3_snapshot_040_filtered.h5", ".csv")
    outdf.to_csv(os.path.join(path_out, outname))

    print(value + " has been processed")
    print(str(len(files) - 1 - index) + " more files to go")
