from PIL import Image, ImageTk
import tkinter
import tkinter.filedialog
import numpy
import cv2
from os import getcwd
from os.path import normpath, basename
import time
import math
from FoxDot import *
import random
from functools import cmp_to_key
from quantize import quantize


global last_frame                                      #creating global variable
last_frame = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
global cap
cap = cv2.VideoCapture("media/fireworks.mp4")


lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

maxCorners = 6
feature_params = dict(maxCorners = maxCorners,
                    qualityLevel = 0.3,
                    minDistance = 10,
                    blockSize = 7 )

# Create some random colors
random_colors = numpy.random.randint(0, 255, (10000, 3))

trajectory_len = 16
detect_interval = 5
trajectories = []
frame_idx = 0

# cap = None
webcam = False
def show_vid():                                        #creating a function
    global trajectories, trajectory_len, frame_idx, detect_interval, prev_gray, frame_gray, cap
    if not cap.isOpened():                             #checks for the opening of camera
        print("cant open the camera")

    # start time to calculate FPS
    start = time.time()

    flag, frame = cap.read()
    frame = cv2.flip(frame, 1)


    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if webcam:
        frame_gray = cv2.flip(frame_gray, 1)
    img = frame.copy()
    if webcam:
        img = cv2.flip(img, 1)
    h, w = img.shape[:2]

    # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        pts0 = numpy.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        pts1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, pts0, None, **lk_params)
        pts0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, pts1, None, **lk_params)
        d = abs(pts0-pts0r).reshape(-1, 2).max(-1)
        good = d < 1
     
        new_trajectories = []

        # Get all the trajectories
        for trajectory, (x, y), good_flag in zip(trajectories, pts1.reshape(-1, 2), good):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            # Newest detected point
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        trajectories = new_trajectories

        # Draw all the trajectories
        for i in range(len(trajectories)):
            cv2.polylines(img, [numpy.int32(trajectories[i])], False, random_colors[i].tolist(), 1)
        # print([numpy.int32(trajectory) for trajectory in trajectories])
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)


    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = numpy.zeros_like(frame_gray)
        mask[:] = 255

        # Lastest point in latest trajectory
        for x, y in [numpy.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            # If good features can be tracked - add that to the trajectories
            for x, y in numpy.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])
    
    frame_idx += 1
    prev_gray = frame_gray

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)

    
    # Show Results
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.imshow('Optical Flow', img)
    # cv2.imshow('Mask', mask)

    if flag is None:
        print("Major error!")
    elif flag:
        global last_frame
        last_frame = img


    pic = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)     #we can change the display color of the frame gray,black&white here
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, show_vid)

def stop_vid():
    global cap
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    root=tkinter.Tk()                                     #assigning root variable for Tkinter as tk
    lmain = tkinter.Label(master=root)
    lmain.grid(column=1, rowspan=4, padx=5, pady=5)
    root.title("Kinetic Soundscapes")            #you can give any title

    # dropdown for scale
    selected_scale = tkinter.StringVar()
    selected_scale.set('major')
    scale_dropdown = tkinter.OptionMenu(root, selected_scale, 'none (atonal)', 'aeolian', 'altered', 'bebopDom', 'bebopDorian', 'bebopMaj', 'bebopMelMin', 'blues', 'chinese', 'chromatic', 'custom', 'default', 'diminished', 'dorian', 'dorian2', 'egyptian', 'freq', 'halfDim', 'halfWhole', 'harmonicMajor', 'harmonicMinor', 'hungarianMinor', 'indian', 'justMajor', 'justMinor', 'locrian', 'locrianMajor', 'lydian', 'lydianAug', 'lydianDom', 'lydianMinor', 'major', 'majorPentatonic', 'melMin5th', 'melodicMajor', 'melodicMinor', 'minMaj', 'minor', 'minorPentatonic', 'mixolydian', 'phrygian', 'prometheus', 'romanianMinor', 'susb9', 'wholeHalf', 'wholeTone', 'yu', 'zhi')
    scale_dropdown.grid(column=0)

    # stop button
    stop_btn = tkinter.Button(root, text="Stop", command=stop_vid, height=3, width=6)
    stop_btn.grid(column=0)

    show_vid()
    root.mainloop()                                  #keeps the application in an infinite loop so it works continuosly
    cap.release()