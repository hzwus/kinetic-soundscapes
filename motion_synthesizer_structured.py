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

# GLOBAL VARIABLES

# motion
default_maxcorners = 10
default_detectinterval = 4
default_trajlen = 20
trajectories = []
lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# music
default_cci = 120
quantized = True
default_bpm = 120
default_melody_synth = 'marimba'
default_accomp_synth = 'ambi'
default_melody_layers = 4
default_accomp_layers = 4
chords = [
            [-14, -12, -10, -7,-5,2, 0,2,3,4,5, 7,9,12], 
            [-10,-8,-6, -3,-1,1,2,3, 4,6,8,9],
            [-11, -9, -7, -4,-2,0, 3, 5, 7]
        ]
chord = chords[0]

# playback
selected_video = ""
cap_exists = False
playing = True
frame_idx = 0
vid_frame = 0
chord_idx = 0
webcam = False
chord_change_interval = default_cci
VIDEO_W = 1000
VIDEO_H = 700

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
            if show_flow:
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        trajectories = new_trajectories

        generate_music(trajectories)

        # Draw all the trajectories
        if show_video == False:
            blank = numpy.zeros_like(img)
            img = blank

        if show_flow == True:
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

    return img


class Sidebar(tkinter.Frame):
    def __init__(self, parent):
        tkinter.Frame.__init__(self, parent, width=0.3*parent.width, height=parent.height, borderwidth=2, padx=5, pady=5, relief='raised')
        self.parent = parent
        self.input_settings = InputSettings(self)
        self.motion_settings = MotionSettings(self)
        self.music_settings = MusicSettings(self)

        self.input_settings.grid(sticky='ew')
        self.input_settings.columnconfigure(0, weight = 1)
        self.input_settings.columnconfigure(1, weight = 1)
        self.motion_settings.grid(stick='ew')
        self.motion_settings.columnconfigure(0, weight = 1)
        self.music_settings.grid(stick='ew')
        self.music_settings.columnconfigure(0, weight = 1)


class InputSettings(tkinter.LabelFrame):
    def __init__(self, parent):
        tkinter.LabelFrame.__init__(self, parent, text="Input")
        self.parent = parent

        self.select_file_button = tkinter.Button(self, text="Select Video File", command=self.select_file, height=2)
        self.select_file_button.grid(column=0, row=0, sticky='we')

        use_webcam_button = tkinter.Button(self, text="Use Webcam", command=self.use_webcam, height=2)
        use_webcam_button.grid(column=1, row=0, sticky='we')

        self.label_text = tkinter.StringVar()
        self.selected_file_label = tkinter.Label(self, textvariable=self.label_text, font=("Helvetica Bold", 14))
        self.selected_file_label.grid(pady=(2,0), columnspan=2, sticky='w')

    def select_file(self):
        global selected_video, webcam
        webcam = False
        # pause_playback()
             
        # file dialog
        root.filename = ""
        selected_video = ""

        self.filename = tkinter.filedialog.askopenfilename(initialdir=getcwd()+'/media', title="Select a video file (mp4)", filetypes=(("mp4 files", "*.mp4"),("all files", "*.*")))
        if self.filename != "":
            selected_video = self.filename
            print(selected_video)
        self.label_text.set(basename(normpath(selected_video)))
        self.focus_force()

        cap = cv2.VideoCapture(self.filename)
        flag, frame = cap.read()
        print("flagk is ", flag)
        process_frame(frame)

        # stop_playback()

    def use_webcam(self):
        print("placeholder for use_webcam()")

class Player(tkinter.Frame):
    def __init__(self, parent, settings):
        tkinter.Frame.__init__(self, parent, width=0.7*parent.width, height=parent.height)
        self.parent = parent
        self.motion_settings = settings.motion_settings

        self.viewer = Viewer(self)
        self.playbar = Playbar(self)

        self.viewer.grid(row=0, column=0)
        self.playbar.grid(row=1, column=0)


class Viewer(tkinter.Frame):
    def __init__(self, parent):
        tkinter.LabelFrame.__init__(self, parent, text="Input")
        self.parent = parent

        self.video_frame = tkinter.Label(self)
        self.video_frame.grid()

        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)


class Playbar(tkinter.Frame):
     def __init__(self, parent):
        tkinter.LabelFrame.__init__(self, parent, text="Input")
        self.parent = parent


class MotionSettings(tkinter.LabelFrame):
    def __init__(self, parent):
        tkinter.LabelFrame.__init__(self, parent, text="Motion Settings")
        self.parent = parent

        self.maxcorners = default_maxcorners

        # Create the labels
        self.maxcorners_label = tkinter.Label(self, text="Max Corners").grid(row=0, column=0, sticky='sw')
        self.trajlen_label = tkinter.Label(self, text="Trajectory Length").grid(row=1, column=0,  sticky='sw')
        self.detectinterval_label = tkinter.Label(self, text="Detect Interval").grid(row=2, column=0, sticky='sw')

        # Create the controls
        self.maxcorners_slider = tkinter.Scale(self, from_=1, to=20, orient=tkinter.HORIZONTAL, resolution = 1, length = 150, sliderlength=20, command=self.slide_maxcorners)
        self.maxcorners_slider.set(default_maxcorners)
        self.trajlen_slider = tkinter.Scale(self, from_=2, to=160, orient=tkinter.HORIZONTAL, resolution = 1, length = 150, sliderlength=20, command=self.slide_trajlen)
        self.trajlen_slider.set(default_trajlen)
        self.detectinterval_slider = tkinter.Scale(self, from_=1, to=50, orient=tkinter.HORIZONTAL, resolution = 1, length = 150, sliderlength=20, command=self.slide_detectinterval)
        self.detectinterval_slider.set(default_detectinterval)

        # Position all elements
        self.maxcorners_slider.grid(row=0, column=1)
        self.trajlen_slider.grid(row=1, column=1)
        self.detectinterval_slider.grid(row=2, column=1)

    def slide_maxcorners(self, var):
        global maxcorners
        maxcorners = self.maxcorners_slider.get()

        # slider for trajectory length
    def slide_trajlen(self, var):
        global trajectory_len
        trajectory_len = self.trajlen_slider.get()

    # slider for detection interval
    def slide_detectinterval(self, var):
        global detect_interval
        detect_interval = self.detectinterval_slider.get()


class MusicSettings(tkinter.LabelFrame):
    def __init__(self, parent):
        tkinter.LabelFrame.__init__(self, parent, text="Music Settings")
        self.parent = parent

        self.root_label = tkinter.Label(self, text="Root").grid(row=0, column=0, sticky='ws')
        self.scale_label = tkinter.Label(self, text="Scale").grid(row=1, column=0, sticky='ws')
        self.tempo_label = tkinter.Label(self, text="Tempo").grid(row=2, column=0, sticky='ws')
        self.cci_label = tkinter.Label(self, text="Chord Change Interval").grid(row=3, column=0, sticky='ws')
        self.melody_synth_label = tkinter.Label(self, text="Melody Synth").grid(row=4, column=0, sticky='ws')
        self.accomp_synth_label = tkinter.Label(self, text="Harmony Synth").grid(row=5, column=0, sticky='ws')
        self.melody_layers_label = tkinter.Label(self, text="Melody Layers").grid(row=6, column=0, sticky='ws')
        self.accomp_layers_label = tkinter.Label(self, text="Harmony Layers").grid(row=7, column=0, sticky='ws')

        # dropdown for root
        self.selected_root = tkinter.StringVar()
        self.selected_root.set('C')
        self.root_dropdown = tkinter.OptionMenu(self, self.selected_root, 'C', 'C#', 'D', 'D#', 'E', 'E#', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'B#', command=self.set_root)
        self.root_dropdown.grid(row=0, column=1, sticky='ew')

        # dropdown for scale
        self.selected_scale = tkinter.StringVar()
        self.selected_scale.set('major')
        scale_dropdown = tkinter.OptionMenu(self, self.selected_scale, 'major', 'minor', 'none (atonal)', 'aeolian', 'altered', 'bebopDom', 'bebopDorian', 'bebopMaj', 'bebopMelMin', 'blues', 'chinese', 'chromatic', 'custom', 'default', 'diminished', 'dorian', 'dorian2', 'egyptian', 'freq', 'halfDim', 'halfWhole', 'harmonicMajor', 'harmonicMinor', 'hungarianMinor', 'indian', 'justMajor', 'justMinor', 'locrian', 'locrianMajor', 'lydian', 'lydianAug', 'lydianDom', 'lydianMinor', 'majorPentatonic', 'melMin5th', 'melodicMajor', 'melodicMinor', 'minMaj', 'minorPentatonic', 'mixolydian', 'phrygian', 'prometheus', 'romanianMinor', 'susb9', 'wholeHalf', 'wholeTone', 'yu', 'zhi', command=self.set_scale)
        scale_dropdown['menu'].insert_separator(3)
        scale_dropdown.grid(row=1, column=1, sticky='ew')

        # slider for tempo
        self.bpm_slider = tkinter.Scale(self, from_=20, to=220, orient=tkinter.HORIZONTAL, resolution = 4, length = 150, sliderlength=20, command=self.slide_bpm)
        self.bpm_slider.set(default_bpm)
        self.bpm_slider.grid(row=2, column=1)

        # slider for chord change interval
        self.cci_slider = tkinter.Scale(self, from_=10, to=500, orient=tkinter.HORIZONTAL, resolution = 5, length = 150, sliderlength=20, command=self.slide_cci)
        self.cci_slider.set(default_cci)
        self.cci_slider.grid(row=3, column=1)

        # dropdown for melody synth
        self.selected_melody_synth = tkinter.StringVar()
        self.selected_melody_synth.set(default_melody_synth)
        # melody_synth = default_melody_synth
        self.melody_synth_dropdown = tkinter.OptionMenu(self, self.selected_melody_synth, 'noise', 'dab', 'varsaw', 'lazer', 'growl', 'bass', 'dirt', 'crunch', 'rave', 'scatter', 'charm', 'bell', 'gong', 'soprano', 'dub', 'viola', 'scratch', 'klank', 'feel', 'glass', 'soft', 'quin', 'pluck', 'spark', 'blip', 'ripple', 'creep', 'orient', 'zap', 'marimba', 'fuzz', 'bug', 'pulse', 'saw', 'snick', 'twang', 'karp', 'arpy', 'nylon', 'donk', 'squish', 'swell', 'razz', 'sitar', 'star', 'jbass', 'piano', 'sawbass', 'prophet', 'pads', 'pasha', 'ambi', 'space', 'keys', 'dbass', 'sinepad', command=self.set_melody_synth)
        self.melody_synth_dropdown.grid(row=4, column=1, sticky='ew')
    
        # dropdown for accomp synth
        self.selected_accomp_synth = tkinter.StringVar()
        self.selected_accomp_synth.set(default_accomp_synth)

        # accomp_synth = default_accomp_synth
        self.accomp_synth_dropdown = tkinter.OptionMenu(self, self.selected_accomp_synth, 'noise', 'dab', 'varsaw', 'lazer', 'growl', 'bass', 'dirt', 'crunch', 'rave', 'scatter', 'charm', 'bell', 'gong', 'soprano', 'dub', 'viola', 'scratch', 'klank', 'feel', 'glass', 'soft', 'quin', 'pluck', 'spark', 'blip', 'ripple', 'creep', 'orient', 'zap', 'marimba', 'fuzz', 'bug', 'pulse', 'saw', 'snick', 'twang', 'karp', 'arpy', 'nylon', 'donk', 'squish', 'swell', 'razz', 'sitar', 'star', 'jbass', 'piano', 'sawbass', 'prophet', 'pads', 'pasha', 'ambi', 'space', 'keys', 'dbass', 'sinepad', command=self.set_accomp_synth)
        self.accomp_synth_dropdown.grid(row=4, column=1, sticky='ew')

        # slider for melody layers
        self.melody_layers_slider = tkinter.Scale(self, from_=0, to=8, orient=tkinter.HORIZONTAL, resolution = 1, length = 150, sliderlength=20, command=self.slide_melody_layers)
        self.melody_layers_slider.set(default_melody_layers)
        self.melody_layers_slider.grid(row=6, column=1)

        # slider for accomp layers
        self.accomp_layers_slider = tkinter.Scale(self, from_=0, to=8, orient=tkinter.HORIZONTAL, resolution = 1, length = 150, sliderlength=20, command=self.slide_accomp_layers)
        self.accomp_layers_slider.set(default_accomp_layers)
        self.accomp_layers_slider.grid(row=7, column=1)


    def set_root(self, var):
        Root.default.set(self.selected_root.get())

    def set_scale(self, var):
        global quantized
        quantized = self.selected_scale.get() != 'none (atonal)'
        Scale.default.set(self.selected_scale.get())

    def slide_bpm(self, var):
        Clock.update_tempo_now(self.bpm_slider.get())

    def slide_cci(self, var):
        global chord_change_interval
        chord_change_interval = self.cci_slider.get()

    def set_melody_synth(self, var):
        global melody_synth
        melody_synth = self.selected_melody_synth.get()

    def set_accomp_synth(self, var):
        global accomp_synth
        accomp_synth = self.selected_accomp_synth.get()
    
    def slide_melody_layers(self, var):
        global melody_layers
        melody_layers = self.melody_layers_slider.get()

    def slide_accomp_layers(self, var):
        global melody_layers
        melody_layers = self.melody_layers_slider.get()


def process_frame(frame):
    frame = image_resize(frame, width=VIDEO_W)
    pic = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     #we can change the display color of the frame gray,black&white here
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    return imgtk

class MainWindow(tkinter.Frame):
    def __init__(self, parent):
        tkinter.Frame.__init__(self, parent)
        self.parent = parent
        self.width= 1280
        self.height = 720
        
        self.synth_dict = {
            'noise': noise, 'dab': dab, 'varsaw': varsaw, 'lazer': lazer, 'growl': growl, 'bass': bass, 'dirt': dirt, 'crunch': crunch, 'rave': rave, 'scatter': scatter, 'charm': charm, 'bell': bell,
            'gong': gong, 'soprano': soprano, 'dub': dub, 'viola': viola, 'scratch': scratch, 'klank': klank, 'feel': feel, 'glass': glass, 'soft': soft, 'quin': quin, 'pluck': pluck, 'spark': spark,
            'blip': blip, 'ripple': ripple, 'creep': creep, 'orient': orient, 'zap': zap, 'marimba': marimba, 'fuzz': fuzz, 'bug': bug, 'pulse': pulse, 'saw': saw, 'snick': snick, 'twang': twang,
            'karp': karp, 'arpy': arpy, 'nylon': nylon, 'donk': donk, 'squish': squish, 'swell': swell, 'razz': razz, 'sitar': sitar, 'star': star, 'jbass': jbass, 'piano': piano, 'sawbass': sawbass,
            'prophet': prophet, 'pads': pads, 'pasha': pasha, 'ambi': ambi, 'space': space, 'keys': keys, 'dbass': dbass, 'sinepad': sinepad
        }

        self.all_accomp = [a1, a2, a3, a4, a5, a6, a7, a8]
        self.all_melody = [m1, m2, m3, m4, m5, m6, m7, m8]

        self.sidebar = Sidebar(self)

        self.sidebar.grid(column=0, row=0, sticky='ns')


        player = tkinter.LabelFrame(self, width=1300, height=1000, borderwidth=2, relief='raised')
        player.grid(column=1, row=0, sticky='ns')

        viewer = tkinter.LabelFrame(player, width=1300, height=900, borderwidth=2, relief='sunken')
        viewer.grid(row=0)

        self.lmain = tkinter.Label(viewer, padx=100,pady=200)
        self.lmain.grid()

        if selected_video != "" or webcam == True:
            self.show_frame()

    def show_frame(self): 
        global h, w
        global playing
        global vid_frame, chord_idx, chord
        global cap, cap_exists
        global trajectories, trajectory_len, frame_idx, detect_interval, prev_gray, frame_gray
        global selected_video
        global webcam


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
        # if flag == False:
        #     stop_playback()
        #     return

        frame = image_resize(frame, width=1000)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if webcam:
            frame_gray = cv2.flip(frame_gray, 1)
        img = frame.copy()
        if webcam:
            img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        img = compute_flow(img)
        
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
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)
        self.lmain.after(10, self.show_frame)







if __name__ == "__main__":
    root = tkinter.Tk()
    root.title("Kinetic Soundscapes")
    
    MainWindow(root).grid()
    root.mainloop()