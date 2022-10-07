
import numpy
import math
import time
import random
from os import getcwd
from os.path import normpath, basename
from functools import cmp_to_key
from util import quantize, image_resize

import cv2
from PIL import Image, ImageTk
import tkinter
import tkinter.filedialog
from FoxDot import *

random.seed(time.time())

players_accomp = [a1, a2, a3, a4, a5, a6, a7, a8]
players_melody = [m1, m2, m3, m4]
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
last_frame = numpy.zeros((480, 720, 3), dtype=numpy.uint8)

global selected_video
selected_video = None

global cap
cap_exists = False

lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

default_maxcorners = 12
default_detectinterval = 4
default_trajlen = 14

default_cci = 120

# Create some random colors
# random_colors = numpy.random.randint(0, 255, (10000, 3))

trajectories = []
frame_idx = 0

# cap = None
webcam = False
chord = chords[0]
chord_change_interval = default_cci
vid_frame = 0
chord_idx = 0

def compute_flow(img):
    global trajectories

    feature_params = dict(maxCorners = max_corners,
                    qualityLevel = 0.3,
                    minDistance = 10,
                    blockSize = 7 )

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
            cv2.polylines(img, [numpy.int32(trajectories[i])], False, (0, 255, 0), 1)
        # print([numpy.int32(trajectory) for trajectory in trajectories])
        # cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)


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

def show_frame(): 
    global h, w
    global playing
    global vid_frame, chord_idx, chord
    global cap, cap_exists
    global trajectories, trajectory_len, frame_idx, detect_interval, prev_gray, frame_gray
    global selected_video

    if cap_exists == False:
        if webcam:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(selected_video)
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
        stop_playback()
        return

    frame = image_resize(frame, width=720)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if webcam:
        frame_gray = cv2.flip(frame_gray, 1)
    img = frame.copy()
    if webcam:
        img = cv2.flip(img, 1)
    h, w = img.shape[:2]

    compute_flow(img)
    
    frame_idx += 1
    prev_gray = frame_gray

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)

    
    # Show Results
    # cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.imshow('Optical Flow', img)
    # cv2.imshow('Mask', mask)


    global last_frame
    last_frame = img

    pic = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)     #we can change the display color of the frame gray,black&white here
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)





if __name__ == '__main__':
    def pause_playback():
        # global cap
        # cap.release()
        # cv2.destroyAllWindows()
        global playing
        playing = False
        play_btn.config(text="Generate")
        Clock.clear()
        root_dropdown.config(state=tkinter.NORMAL)
        scale_dropdown.config(state=tkinter.NORMAL)

    def start_playback():
        global playing, selected_video
        playing = True
        play_btn.config(text="Pause")
        root_dropdown.config(state=tkinter.DISABLED)
        scale_dropdown.config(state=tkinter.DISABLED)

        show_frame()

    def stop_playback():
        global cap, cap_exists, selected_video, trajectories
        pause_playback()
        play_btn.config(text="Generate")
        cap = cv2.VideoCapture(selected_video)

        flag, frame = cap.read()
        frame = image_resize(frame, width=720)
        pic = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     #we can change the display color of the frame gray,black&white here
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        trajectories = []

        cap.release()
        cap_exists = False
        
    root=tkinter.Tk()                                     
    root.title("Kinetic Soundscapes")            #you can give any title

    sidebar = tkinter.LabelFrame(root, width=280, height=720, borderwidth=2, padx=5, pady=5, relief='raised')
    sidebar.grid(column=0, sticky='ns')

     # file dialog
    root.filename = ""
    selected_video = ""
    var = tkinter.StringVar()

    def select_file():
        global selected_video, cap_exists
        pause_playback()
        root.filename = tkinter.filedialog.askopenfilename(initialdir=getcwd()+'/media', title="Select a video file (mp4)", filetypes=(("mp4 files", "*.mp4"),("all files", "*.*")))
        selected_video = root.filename
        print("selected_video is ", selected_video)
        var.set(basename(normpath(selected_video)))

        cap = cv2.VideoCapture(root.filename)
        flag, frame = cap.read()
        frame = image_resize(frame, width=720)
        pic = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     #we can change the display color of the frame gray,black&white here
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        stop_playback()

    select_file_button = tkinter.Button(sidebar, text="Select Video File", command=select_file, height=2)
    select_file_button.grid(sticky='ew')

    selected_file_label = tkinter.Label(sidebar, textvariable=var, font=("Helvetica Bold", 14))
    selected_file_label.grid(pady=(2,0))


    sidebar_motion = tkinter.LabelFrame(sidebar, text="Motion Settings", width=280, height=360, padx=5, pady=5)
    sidebar_motion.grid(sticky='ew', pady=5)

    sidebar_motion.columnconfigure(0, weight = 1)

    maxcorners_label = tkinter.Label(sidebar_motion, text="Max Corners").grid(row=0, column=0, sticky='sw')
    trajlen_label = tkinter.Label(sidebar_motion, text="Trajectory Length").grid(row=1, column=0,  sticky='sw')
    detectinterval_label = tkinter.Label(sidebar_motion, text="Detect Interval").grid(row=2, column=0, sticky='sw')

    # slider for max corners
    def slide_maxcorners(var):
        global max_corners, default_maxcorners
        max_corners = maxcorners_slider.get()
    maxcorners_slider = tkinter.Scale(sidebar_motion, from_=2, to=20, orient=tkinter.HORIZONTAL, resolution = 1, length = 150, sliderlength=20, command=slide_maxcorners)
    maxcorners_slider.set(default_maxcorners)
    maxcorners_slider.grid(row=0, column=1)

    # slider for trajectory length
    def slide_trajlen(var):
        global trajectory_len, default_trajlen
        trajectory_len = trajlen_slider.get()
    trajlen_slider = tkinter.Scale(sidebar_motion, from_=2, to=160, orient=tkinter.HORIZONTAL, resolution = 1, length = 150, sliderlength=20, command=slide_trajlen)
    trajlen_slider.set(default_trajlen)
    trajlen_slider.grid(row=1, column=1)

    # slider for detection interval
    def slide_detectinterval(var):
        global detect_interval, default_detectinterval
        detect_interval = detectinterval_slider.get()
    detectinterval_slider = tkinter.Scale(sidebar_motion, from_=1, to=50, orient=tkinter.HORIZONTAL, resolution = 1, length = 150, sliderlength=20, command=slide_detectinterval)
    detectinterval_slider.set(default_detectinterval)
    detectinterval_slider.grid(row=2, column=1)

    def reset_motion_settings():
        global default_maxcorners, default_trajlen, default_detectinterval
        maxcorners_slider.set(default_maxcorners)
        trajlen_slider.set(default_trajlen)
        detectinterval_slider.set(default_detectinterval)
    reset_motion_btn = tkinter.Button(sidebar_motion, text="Reset", command=reset_motion_settings)
    reset_motion_btn.grid(columnspan=2)

    sidebar_music = tkinter.LabelFrame(sidebar, text="Music Settings", width=280, height=360, padx=5, pady=5)
    sidebar_music.grid(sticky='ew', pady=5)
    sidebar_music.columnconfigure(0, weight = 1)

    player = tkinter.LabelFrame(root, width=1000, height=720, borderwidth=2, relief='raised')
    player.grid(column=1, row=0)

    viewer = tkinter.LabelFrame(player, width=1000, height=700, borderwidth=2, relief='sunken')
    viewer.grid(row=0)

    playbar = tkinter.LabelFrame(player, width=1000, height=70, borderwidth=0)
    playbar.grid(row=1)

    lmain = tkinter.Label(viewer)
    lmain.grid()

    img = Image.fromarray(numpy.zeros((480,720,3), numpy.uint8))
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    root_label = tkinter.Label(sidebar_music, text="Root").grid(row=0, column=0, sticky='ws')
    scale_label = tkinter.Label(sidebar_music, text="Scale").grid(row=1, column=0, sticky='ws')
    tempo_label = tkinter.Label(sidebar_music, text="Tempo").grid(row=2, column=0, sticky='ws')
    cci_label = tkinter.Label(sidebar_music, text="Chord Change Interval").grid(row=3, column=0, sticky='ws')

    # dropdown for root
    def set_root(var):
        Root.default.set(selected_root.get())
        print("setting root to ", Root.default.char)
    selected_root = tkinter.StringVar()
    selected_root.set('C')
    root_dropdown = tkinter.OptionMenu(sidebar_music, selected_root, 'C', 'C#', 'D', 'D#', 'E', 'E#', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'B#', command=set_root)
    root_dropdown.grid(row=0, column=1, sticky='ew')
 

    # dropdown for scale
    def set_scale(var):
        global quantized
        if selected_scale.get() == 'none (atonal)':
            quantized = False
        else:
            quantized = True
            Scale.default.set(selected_scale.get())
        print("setting scale to ", Scale.default.name)
    selected_scale = tkinter.StringVar()
    selected_scale.set('major')
    scale_dropdown = tkinter.OptionMenu(sidebar_music, selected_scale, 'major', 'minor', 'none (atonal)', 'aeolian', 'altered', 'bebopDom', 'bebopDorian', 'bebopMaj', 'bebopMelMin', 'blues', 'chinese', 'chromatic', 'custom', 'default', 'diminished', 'dorian', 'dorian2', 'egyptian', 'freq', 'halfDim', 'halfWhole', 'harmonicMajor', 'harmonicMinor', 'hungarianMinor', 'indian', 'justMajor', 'justMinor', 'locrian', 'locrianMajor', 'lydian', 'lydianAug', 'lydianDom', 'lydianMinor', 'majorPentatonic', 'melMin5th', 'melodicMajor', 'melodicMinor', 'minMaj', 'minorPentatonic', 'mixolydian', 'phrygian', 'prometheus', 'romanianMinor', 'susb9', 'wholeHalf', 'wholeTone', 'yu', 'zhi', command=set_scale)
    scale_dropdown['menu'].insert_separator(3)
    scale_dropdown.grid(row=1, column=1, sticky='ew')

    # slider for tempo
    def slide_bpm(var):
        Clock.update_tempo_now(bpm_slider.get())
    bpm_slider = tkinter.Scale(sidebar_music, from_=20, to=220, orient=tkinter.HORIZONTAL, resolution = 4, length = 150, sliderlength=20, command=slide_bpm)
    bpm_slider.set(120)
    bpm_slider.grid(row=2, column=1)

    # slider for chord change interval
    def slide_cci(var):
        global chord_change_interval
        chord_change_interval = cci_slider.get()
    cci_slider = tkinter.Scale(sidebar_music, from_=10, to=500, orient=tkinter.HORIZONTAL, resolution = 5, length = 150, sliderlength=20, command=slide_cci)
    cci_slider.set(default_cci)
    cci_slider.grid(row=3, column=1)

    # play/pause button
    def switch():
        global playing
        if playing:
            pause_playback()
        else:
            if selected_video != "":
                start_playback()

    play_btn = tkinter.Button(playbar, text="Generate", command=switch, height=3, width=6, relief='raised')
    play_btn.grid(column=0, row=0, pady=5)


    # stop button
    stop_btn = tkinter.Button(playbar, text="Stop", command=stop_playback, height=3, width=6, relief='raised')
    stop_btn.grid(column=1, row=0, pady=5)
    
    if selected_video != "":
        show_frame()

    root.mainloop()                                  #keeps the application in an infinite loop so it works continuosly