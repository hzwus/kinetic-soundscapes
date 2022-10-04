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

random.seed(time.time())

players_accomp = [a1, a2, a3, a4, a5, a6, a7, a8]
players_melody = [m1, m2, m3]
synths = [ambi, sinepad]
chords = [
            [-14, -12, -10, -7,-5,2, 0,2,3,4,5, 7,9,12], 
            [-10,-8,-6, -3,-1,1,2,3, 4,6,8,9],
            [-11, -9, -7, -4,-2,0, 3, 5, 7]
            ]
max_players = len(players_accomp) + len(players_melody)

quantized = True
playing = False
global last_frame                                      #creating global variable
last_frame = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
global cap

cap = cv2.VideoCapture("media/fireworks.mp4")
cap_exists = True

lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

maxCorners = 8
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
chord = chords[0]
chord_change_interval = 120
vid_frame = 0
chord_idx = 0

def generate_music(trajectories):
    melody_attrs = [None] * len(players_melody)
    accomp_attrs = [None] * len(players_accomp)
        
    # print(len(trajectories))
    if len(trajectories) > max_players:
        trajectories_sorted = sorted(trajectories, key=cmp_to_key(lambda t1, t2: math.dist(t2[0], t2[-1]) - math.dist(t1[0], t1[-1])))

        trajectories_best = trajectories_sorted[:max_players]
        # trajectories_best = random.sample(trajectories, max_players)
        # for t1 in trajectories_best:
        #     print(math.dist(t1[0], t1[-1]))
        # print("DONE")
    else:
        trajectories_best = trajectories

    for i in range(len(trajectories_best)):
        if i > len(players_accomp)-1:
            t = trajectories_best[i]
            mag = math.dist(t[0], t[-1])
            vol = mag / 200
            # print(vol)
            pitch = (h - t[-1][1]) / h * 24 - 12
            if quantized:
                pitch = round(pitch)
            dur = 1/3
            pan = (t[-1][0] / w) * 1.6 - 0.8
            melody_attrs[i-len(players_accomp)] = (pitch, vol, dur, pan)
        else:
            t = trajectories_best[i]
            mag = math.dist(t[0], t[-1])
            vol = mag / 200
            # print(vol)
            pitch = (h - t[-1][1]) / h * 24 - 12
            if quantized:
                pitch = round(pitch)
            dur = 1/3
            pan = (t[-1][0] / w) * 1.6 - 0.8
            accomp_attrs[i] = (pitch, vol, dur, pan)

        # print(mag)
    # print(player_attrs)
    
    for i in range(len(melody_attrs)):
        if melody_attrs[i] is None:
            break
        pitch = melody_attrs[i][0]
        # pitch = quantize(pitch, [-7,-5,2, 0,2,3,4,5, 7,9,12])
        # print(pitch)
        # pitch = quantize(pitch, chord)
    
        vol = melody_attrs[i][1]
        dur = melody_attrs[i][2]
        pan = melody_attrs[i][3]
        delay = random.choice([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
        # print(delay)

        synth_rand = random.choice(synths)
        players_melody[i] >> marimba(pitch, dur=1/4, amp=min(0.5, vol), pan=pan, room=0.5, mix=0.2, sus=1, delay=0)

    
    for i in range(len(accomp_attrs)):
        if accomp_attrs[i] is None:
            break
        pitch = accomp_attrs[i][0]
        # pitch = quantize(pitch, [-7,-5,2, 0,2,3,4,5, 7,9,12])
        # print(pitch)
        if quantized:
            pitch = quantize(pitch, chord)
        # pitch = (pitch, pitch+2, pitch+4)
    
        vol = accomp_attrs[i][1]
        dur = accomp_attrs[i][2]
        pan = accomp_attrs[i][3]
        # delay = random.choice([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
        # print(delay)

        synth_rand = random.choice(synths)
        players_accomp[i] >> synth_rand(pitch, dur=5, amp=min(0.2, vol), pan=pan, room=0.5, mix=0.2, sus=8, delay=0)

def show_vid(): 
    global h, w
    global playing
    global vid_frame, chord_idx, chord
    global cap, cap_exists
    global trajectories, trajectory_len, frame_idx, detect_interval, prev_gray, frame_gray

    if cap_exists == False:
        cap = cv2.VideoCapture("media/fireworks.mp4")
        cap_exists = True

    vid_frame += 1
    if vid_frame % chord_change_interval == 0:
        chord_idx += 1
        if chord_idx == len(chords):
            chord_idx = 0
        chord = chords[chord_idx]
        print("CHORD IS ", chord_idx)
    # print(playing)
    if playing == False:   
        return                                    #creating a function
    if not cap.isOpened():                             #checks for the opening of camera
        print("cant open the camera")

    # start time to calculate FPS
    start = time.time()

    flag, frame = cap.read()
    if flag == False:
        pause_playback()
        img = Image.fromarray(numpy.zeros((h,w,3), numpy.uint8))
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        play_btn.config(text="Play")
        cap_exists = False
        

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

        generate_music(trajectories)

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


    global last_frame
    last_frame = img


    pic = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)     #we can change the display color of the frame gray,black&white here
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_vid)

    return trajectories

def pause_playback():
    # global cap
    # cap.release()
    # cv2.destroyAllWindows()
    global playing
    playing = False
    play_btn.config(text="Play")
    Clock.clear()

def start_playback():
    global playing
    playing = True
    play_btn.config(text="Pause")
    show_vid()

def stop_playback():
    global cap, cap_exists
    pause_playback()
    play_btn.config(text="Play")
    cap.release()
    img = Image.fromarray(numpy.zeros((h,w,3), numpy.uint8))
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    cap_exists = False



if __name__ == '__main__':
    root=tkinter.Tk()                                     #assigning root variable for Tkinter as tk
    lmain = tkinter.Label(master=root)
    lmain.grid(column=1, rowspan=4, padx=5, pady=5)
    root.title("Kinetic Soundscapes")            #you can give any title

    root.geometry('1280x720')


    # file dialog
    root.filename = "default"
    def select_file():
        root.filename = tkinter.filedialog.askopenfilename(initialdir=getcwd(), title="Select a video file (mp4)", filetypes=(("mp4 files", "*.mp4"),("all files", "*.*")))
        cap = cv2.VideoCapture(root.filename)
        flag, frame = cap.read()
        pic = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     #we can change the display color of the frame gray,black&white here
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
    select_file_button = tkinter.Button(root, text="Select Video File", command=select_file)
    select_file_button.grid(column=0)
    # print("filename is ", root.filename)
    # cap = cv2.VideoCapture(root.filename)



    # var = tkinter.StringVar()
    # selected_file_label = tkinter.Label(root, textvariable=var)
    # var.set(basename(normpath(root.filename)))
    # selected_file_label.grid(column=0)

    # dropdown for scale
    def set_scale(var):
        Scale.default.set(selected_scale.get())
        print("setting scale to ", Scale.default.name)
    selected_scale = tkinter.StringVar()
    selected_scale.set('major')
    scale_dropdown = tkinter.OptionMenu(root, selected_scale, 'none (atonal)', 'aeolian', 'altered', 'bebopDom', 'bebopDorian', 'bebopMaj', 'bebopMelMin', 'blues', 'chinese', 'chromatic', 'custom', 'default', 'diminished', 'dorian', 'dorian2', 'egyptian', 'freq', 'halfDim', 'halfWhole', 'harmonicMajor', 'harmonicMinor', 'hungarianMinor', 'indian', 'justMajor', 'justMinor', 'locrian', 'locrianMajor', 'lydian', 'lydianAug', 'lydianDom', 'lydianMinor', 'major', 'majorPentatonic', 'melMin5th', 'melodicMajor', 'melodicMinor', 'minMaj', 'minor', 'minorPentatonic', 'mixolydian', 'phrygian', 'prometheus', 'romanianMinor', 'susb9', 'wholeHalf', 'wholeTone', 'yu', 'zhi', command=set_scale)
    scale_dropdown.grid(column=0, row=1)

    # slider for tempo
    def slide_bpm(var):
        global selected_bpm 
        Clock.update_tempo_now(bpm_slider.get())
    bpm_slider = tkinter.Scale(root, from_=20, to=220, orient=tkinter.HORIZONTAL, resolution = 4, length = 200, sliderlength=20, command=slide_bpm)
    bpm_slider.set(120)
    bpm_slider.grid(column=0, row=2)

    # play/pause button
    def switch():
        global playing
        if playing:
            pause_playback()
        else:
            start_playback()

    play_btn = tkinter.Button(root, text="Play", command=switch, height=3, width=6)
    play_btn.grid(column=0, row=9)

    # stop button
    stop_btn = tkinter.Button(root, text="Stop", command=stop_playback, height=3, width=6)
    stop_btn.grid(column=0, row=10)
    
    show_vid()
    root.mainloop()                                  #keeps the application in an infinite loop so it works continuosly
    cap.release()