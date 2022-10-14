
import numpy as np
import math
import statistics
import time
import random
from os import getcwd
from os.path import normpath, basename
from functools import cmp_to_key
from util import quantize, image_resize
import colorsys


import cv2
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.filedialog
import FoxDot as fd

random.seed(time.time())

synth_dict = {
    'noise': fd.noise, 'dab': fd.dab, 'varsaw': fd.varsaw, 'lazer': fd.lazer, 'growl': fd.growl, 'bass': fd.bass, 'dirt': fd.dirt, 'crunch': fd.crunch, 'rave': fd.rave, 'scatter': fd.scatter, 'charm': fd.charm, 'bell': fd.bell,
    'gong': fd.gong, 'soprano': fd.soprano, 'dub': fd.dub, 'viola': fd.viola, 'scratch': fd.scratch, 'klank': fd.klank, 'feel': fd.feel, 'glass': fd.glass, 'soft': fd.soft, 'quin': fd.quin, 'pluck': fd.pluck, 'spark': fd.spark,
    'blip': fd.blip, 'ripple': fd.ripple, 'creep': fd.creep, 'orient': fd.orient, 'zap': fd.zap, 'marimba': fd.marimba, 'fuzz': fd.fuzz, 'bug': fd.bug, 'pulse': fd.pulse, 'saw': fd.saw, 'snick': fd.snick, 'twang': fd.twang,
    'karp': fd.karp, 'arpy': fd.arpy, 'nylon': fd.nylon, 'donk': fd.donk, 'squish': fd.squish, 'swell': fd.swell, 'razz': fd.razz, 'sitar': fd.sitar, 'star': fd.star, 'jbass': fd.jbass, 'piano': fd.piano, 'sawbass': fd.sawbass,
    'prophet': fd.prophet, 'pads': fd.pads, 'pasha': fd.pasha, 'ambi': fd.ambi, 'space': fd.space, 'keys': fd.keys, 'dbass': fd.dbass, 'sinepad': fd.sinepad
}

# motion
lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

default_maxcorners = 12
default_detectinterval = 3
default_trajlen = 30
trajectories = []

# music
all_accomp = [fd.a1, fd.a2, fd.a3, fd.a4, fd.a5, fd.a6, fd.a7, fd.a8]
all_melody = [fd.m1, fd.m2, fd.m3, fd.m4, fd.m5, fd.m6, fd.m7, fd.m8]

default_melody_layers = 2
default_accomp_layers = 4

default_melody_synth = 'piano'
default_accomp_synth = 'piano'

# default_cci = 120
default_bpm = 120
quantized = True

default_harmony_sus = 5

# histogram for binning detected color values
color_hist = np.zeros(4)
# based on Scriabin's color to tone mapping
# chords = [
#     [-14,-12,-10,-7,-5,-3,0,2,4,7,9,11], # C
#     [-10,-8,-6,-3,-1,1,4,6,8,11,13,15],  # G
#     [-13,-11,-9,-6,-4,-2,1,3,5,8,10,12],  # D
#     [-16,-14,-12,-9,-7,-5,-2,0,2,5,7,9,12,14,16], # A
#     [-12,-10,-8,-5,-3,-1,2,4,6,9,11,13],  # E
#     [-15,-13,-11,-8,-6,-4,-1,1,3,6,8,10],  # B
#     [-11.5,-10,-7.5,-4.5,-3,-1.5,3.5,5,7.5,10.5,12,14.5], # F#
#     [-13.5,-12,-9.5,-6.5,-5,-2.5,0.5,2,4.5,7.5,9,11.5,14.5,16,18.5], # C#
#     [-10.5,-8,-6.5,-3.5,-1,0.5,4.5,7,8.5,11.5,14,15.5], # A♭
#     [-13.5,-11,-9.5,-6.5,-4,-2.5,1.5,4,5.5,8.5,11,12.5], # E♭
#     [-15.5,-13,-11,-8.5,-6,-4,-1.5,1,3,5.5,8,10,12.5,15,17], # B♭
#     [-11,-9,-7,-4,-2,0,3,5,7,10,12,14] # F
# ]

# chords = [
#             [-14, -12, -10, -7,-5,2, 0,2,3,4,5, 7,9,12], 
#             [-10,-8,-6, -3,-1,1,2,3, 4,6,8,9],
#             [-11, -9, -7, -4,-2,0, 3, 5, 7]
#             ]
base_chord = [-14,-12,-10,-7,-5,-3,0,2,4,7,9,11,14,16,18]
# chord_change_interval = default_cci
prev_melody_pitches = []

chords = {
    "none": [0, 0, 0, 0],
    "cinematic I VI III IV": [0, 8, 4, 5], # I VI III IV
    "basic I IV V I": [0, 5, 7, 0] # I IV V I
}
default_chord_setting = "cinematic I VI III IV"
chord = default_chord_setting
chord_setting = default_chord_setting

# cinematic
roots = [0, 8, 4, 5]  # I VI III IV
# # basic
# roots = ['C', 'G', 'F', 'D']

# foxdot roots = [C - 0, C# - 1, D - 2, D# - 3 E - 4, F - 5, F# - 6, G - 7, Ab - 8, A - 9, Bb - 10, B - 11]
base_roots = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F']
base_root = 'C'

#playback
playing = False
show_video = True
show_flow = True
selected_video = None
cap_exists = False
frame_idx = 0
webcam = False
vid_frame = 0
chord_idx = 0
last_picked_color = 0
VIDEO_W = 1000
VIDEO_H = 700

global last_frame                                      #creating global variable
# last_frame = np.zeros((480, 720, 3), dtype=np.uint8)

def convert_for_tk(frame):
    frame = image_resize(frame, width=VIDEO_W)
    pic = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     #we can change the display color of the frame gray,black&white here
    img = Image.fromarray(pic)
    return ImageTk.PhotoImage(image=img)
    
def compute_flow(img):
    global trajectories

    original_img = img.copy()

    feature_params = dict(maxCorners = max_corners,
                    qualityLevel = 0.3,
                    minDistance = 10,
                    blockSize = 7 )

    # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        pts0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
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

        # mark the indices of the trajectories that are being used to play notes
        musical_trajectories = generate_music(trajectories, original_img)

        trajectories_melody = musical_trajectories[:melody_layers]
        trajectories_accomp = musical_trajectories[melody_layers:]

        # Draw all the trajectories
        if show_video == False:
            blank = np.zeros_like(img)
            img = blank

        if show_flow == True:
            if show_video == False:
                alpha = 1.0
            else:
                alpha = 0.65
            overlay = img.copy()
            for i in range(len(trajectories)):
                if i in trajectories_melody:
                    cv2.polylines(overlay, [np.int32(trajectories[i])], False, (0, 187, 255), 5)
                elif i in trajectories_accomp:
                    cv2.polylines(overlay, [np.int32(trajectories[i])], False, (255, 187, 0), 5)
                else:
                    cv2.polylines(overlay, [np.int32(trajectories[i])], False, (255, 255, 255), 1)
            new_img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
            img = new_img
            # print([np.int32(trajectory) for trajectory in trajectories])
            # cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)


    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Lastest point in latest trajectory
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            # If good features can be tracked - add that to the trajectories
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    return img

def compute_path_dist(trajectory):
    dist = 0
    for i in range(1, len(trajectory)):
        dist += math.dist(trajectory[i], trajectory[i-1])
    return dist

# compute volume based on speed of flow, calibrated according to type of synth, pitch, and flag (melody=0, accomp=1)
def compute_volume(speed, pitch, synth, flag, layers):
    vol = speed
    if flag == 0:
        if synth == "sinepad":         # fix issue of high notes sounding way louder than low notes for sinepad
            vol *= 3
            vol /= (pitch+30)
        elif synth in ("marimba", "gong", "keys", "scatter"):
            vol *= 1.2
        elif synth in ("bell", "sitar", "karp", "space", "pluck"):
            vol *= 0.1
        elif synth in ("nylon"):
            vol *= 0.05
        else:
            vol *= 0.5

    elif flag == 1:
        if synth == "sinepad":         # fix issue of high notes sounding way louder than low notes for sinepad
            vol *= 3
            vol /= (pitch+30)
        elif synth in ("nylon", "pulse", "saw", "bug", "creep", "bell", "ripple"):
            vol *= 0.03
        elif synth == "glass":
            vol *= 0.3
        elif synth in ("klank", "charm", "ambi"):
            vol *= 0.2
        elif synth in ("gong", "keys"):
            vol *= 0.7
        elif synth in ("piano"):
            vol *= 0.5
        else:
            vol *= 0.05

    # account for faster motion from webcam input
    if webcam == True:
        vol /= 2

    # if layers > 4:
    #     vol *= 6 / layers

    # print(vol)
    return vol

# reduce the instance of the same notes repeating
def differentiate_pitches(pitches, prev_pitches):
    # print("prev pitches:", prev_pitches)
    # print("current pitches:", pitches)
    differentiated = pitches
    for i in range(len(pitches)):
        if pitches[i] is None or prev_pitches[i] is None:
            continue
        if pitches[i] == prev_pitches[i]:
            # if pitches[i] % 12 in (0, 2, 4, 6):
            #     continue
            roll = random.uniform(0, 1)
            if roll > 0.25:
                differentiated[i] += random.choice([1, -1])
    # print("after diff:", differentiated)
    return differentiated

# given a set of pitches, adjust them so that they sound more harmonious (make them all even or odd)
def refine_pitches(pitches):
    refined_pitches = pitches
    x = random.choice(['even', 'even', 'odd'])
    if x == 'even':
        for i in range(len(refined_pitches)):
            pitch = refined_pitches[i]
            if pitch is None:
                continue
            if pitch % 2 != 0:
                refined_pitches[i] += random.choice([1, -1])
    elif x == 'odd':
        for i in range(len(refined_pitches)):
            pitch = refined_pitches[i]
            if pitch is None:
                continue
            if pitch % 2 == 0:
                refined_pitches[i] += random.choice([1, -1, 0, 0])
    return refined_pitches

# probabilistically choose the next chord based on the circle of fifths
def choose_next_chord(chord_idx):
    print("current chord_idx is ", chord_idx)
    probabilities = [0] * 12
    for i in range(12):
        if i != chord_idx:
            distance = abs(chord_idx - i) % 12
            prob = 12 / distance
            probabilities[i] = prob

    print("probabilities", probabilities)

    choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    next_chord = random.choices(choices, weights=probabilities, k=1)
    print("next chord is ", next_chord[0])
    return next_chord[0]
        
# add a level of dissonance to the chord depending on how much color variety there is in the image
def add_dissonance(chord, dissonance):
    modified_chord = chord.copy()
    threshold = dissonance / 5
    for i in range(len(modified_chord)):
        #[0,2,4]
        roll = random.uniform(0, 1)
        # print("roll, threshold: ", roll, threshold)
        if roll < threshold:
            modified_chord.append(random.choice([chord[i]+1, chord[i]-1]))

    # print("modified chord is ", modified_chord)
    return modified_chord

def generate_music(trajectories, img):
    global melody_layers, accomp_layers, melody_synth, accomp_synth, vid_frame, color_hist, chord, prev_melody_pitches, harmony_sus, chord_setting

    players_accomp = all_accomp[:accomp_layers]
    players_melody = all_melody[:melody_layers]

    max_players = len(players_accomp) + len(players_melody)

    melody_attrs = [None] * len(players_melody)
    accomp_attrs = [None] * len(players_accomp)

    if melody_layers == 0:
        for melody in all_melody:
            melody.stop()
    if accomp_layers == 0:
        for accomp in all_accomp:
            accomp.stop()

        
    # print(len(trajectories))
    trajectories_best_indices = []
    if len(trajectories) > max_players:
        # trajectories_sorted = sorted(trajectories, key=cmp_to_key(lambda t1, t2: math.dist(t2[0], t2[-1]) - math.dist(t1[0], t1[-1])))

        trajectories_sorted = [(trajectory, idx) for idx, trajectory in sorted(enumerate(trajectories), key=cmp_to_key(lambda t1, t2: math.dist(t2[1][0], t2[1][-1]) - math.dist(t1[1][0], t1[1][-1])))]
        trajectories_best = [trajectory[0] for trajectory in trajectories_sorted][:max_players]
        trajectories_best_indices = [trajectory[1] for trajectory in trajectories_sorted][:max_players]
        # print("BEST INDICES", trajectories_best_indices)
        # trajectories_best = random.sample(trajectories, max_players)
        # for t1 in trajectories_best:
        #     print(math.dist(t1[0], t1[-1]))
        # print("DONE")
    else:
        trajectories_best = trajectories

    # map variation in direction to variation in note length
    # but make sure to account for perspective (star wars hyperspace style motion would result in high variation of direction)
    # one possible way: split the screen into four quadrants and calculate the stdev in each quadrant. then average them
    #                   if a quadrant doesn't have enough data points, ignore it
    quadrants = [ [], [], [], [] ]
    mag_total = 0
    for t in trajectories_best:
        final_x, final_y = t[-1][0], t[-1][1]
        initial_x, initial_y = t[-2][0], t[-2][1]

        fx = final_x - initial_x
        fy = final_y - initial_y
        mag = math.dist(t[0], t[-1])
        mag_total += mag

        # direction of flow weighted by magnitude
        dir = (np.arctan2(fy, fx) + np.pi) * mag

        if final_x < w/2:
            if final_y < h/2: # SW quadrant
                quadrants[0].append(dir)
            else: # NW quadrant
                quadrants[1].append(dir)
        else:
            if final_y < h/2: # SE quadrant
                quadrants[2].append(dir)
            else: # NE quadrant
                quadrants[3].append(dir)
     

    overall_stdev = 0
    for quadrant in quadrants:
        if len(quadrant) >= 2 and mag_total > 0:
            quadrant = [val / mag_total for val in quadrant]     # normalize weight based on total magnitude
            quadrant_stdev = statistics.stdev(quadrant)
            weighted_stdev = (len(quadrant)/len(trajectories))*quadrant_stdev
            overall_stdev += weighted_stdev
    overall_stdev *= 10
    print("overall std dev of directions is ", overall_stdev)

    avg_speed = 0
    for i in range(len(trajectories)):
        t = trajectories[i]
        dist = compute_path_dist(t)
        speed = dist / (len(t))
        avg_speed += speed
    if len(trajectories) > 0:
        avg_speed /= len(trajectories)
    # print("avg speed of trajectories is ", avg_speed)
    tempo = 110*math.log(0.3*avg_speed+1.5) - 20
    fd.Clock.update_tempo_now(round(tempo/10)*10)
    # print("tempo is ", fd.Clock.bpm)


    melody_note_lengths = [0.25]
    if overall_stdev > 0.05:
        melody_note_lengths.append(0.5)
    if overall_stdev > 0.1:
        melody_note_lengths.append(0.25)
    if overall_stdev > 0.25:
        melody_note_lengths.append(1)
    if overall_stdev > 0.5:
        melody_note_lengths.append(0.75)
    if overall_stdev > 1:
        melody_note_lengths.extend([1, 0.75, 0.5, 0.5, 0.25, 0.25])
    if overall_stdev > 2:
        melody_note_lengths.extend([1/3, 3/4, 1])

    accomp_note_lengths = [4]
    if overall_stdev > 0.05:
        accomp_note_lengths.append(2)
    if overall_stdev > 0.1:
        accomp_note_lengths.append(1)
    if overall_stdev > 0.25:
        accomp_note_lengths.append(0.5)
    if overall_stdev > 0.5:
        accomp_note_lengths.append(0.5)
    if overall_stdev > 1:
        accomp_note_lengths.extend([0.5, 1, 2])
    if overall_stdev > 2:
        accomp_note_lengths.extend([0.25, 2, 3])

    for i in range(len(trajectories)):
        t = trajectories[i]
        loc_x = int(t[-1][0])
        loc_y = int(t[-1][1])

        if loc_x >= w: 
            loc_x = w-1
        elif loc_x < 0:
            loc_x = 0
        if loc_y >= h:
            loc_y = h-1
        elif loc_y < 0:
            loc_y = 0
        # bgr_color = img[loc_y, loc_x]
        # print("bgr: ", bgr_color)
        b, g, r = img[loc_y, loc_x] / 255
        color = colorsys.rgb_to_hsv(r, g, b)
        hue = color[0]
        saturation = color[1]
        value = color[2]
        # print("hsv:", color)
        # print("hue: ", hue)
        if saturation < 0.1 or value < 0.1:
            # print("not enough color")
            continue
        else:
            # print("hue is ", hue)
            color_hist[math.floor(hue * 4)] += 1

    # detect color in 10 frame intervals
    if vid_frame % 10 == 0 and np.sum(color_hist) > 10:
        
        # compute the most prominent color to assign a chord
       
        # print(color_hist)
        color_hist_indices = []
        for idx in range(len(color_hist)):
            for i in range(int(color_hist[idx])):
                color_hist_indices.append(idx)
        # print("color_hist_indices:", color_hist_indices)
        color_variation = np.std(color_hist_indices)
        # print("color variation level:", color_variation)
        picked_color = np.argmax(color_hist)
        # print("color picked is ", picked_color)

        fd.Root.default.set((base_roots.index(base_root) + chords[chord_setting][picked_color]) % 12)
        # print("chord chosen is ", chords[chord_idx])
        # chord = chords[chord_idx]

        # print("before dissonance: ", chord)
        # chord = add_dissonance(base_chord, color_variation)
        # print("after dissonance: ", chord)

        color_hist = np.zeros(4)
    vid_frame += 1

    for i in range(len(trajectories_best)):
        t = trajectories_best[i]
        dist = compute_path_dist(t)
        speed = dist / (len(t))
        pan = (t[-1][0] / w) * 1.8 - 0.9

        if i > len(players_accomp)-1:
            dur = random.choice(melody_note_lengths)
            # pitches = []
            # for j in range(len(t)):
            #     pitch = (h - t[j][1]) / h * 24 - 6
            #     if quantized:
            #         pitch = round(pitch)
            #     pitches.append(pitch)
            pitch = (h - t[-1][1]) / h * 27 - 12
            if quantized:
                pitch = round(pitch)
            vol = compute_volume(speed, pitch, melody_synth, 0, melody_layers)
            lpf = 1500*avg_speed + 300
            # print("lpf is ", lpf)
            melody_attrs[i-len(players_accomp)] = (pitch, vol, dur, pan, lpf)
        else:
            pitch = (h - t[-1][1]) / h * 21 - 12
            if quantized:
                pitch = round(pitch)
            vol = compute_volume(speed, pitch, accomp_synth, 1, accomp_layers)
            lpf = 1500*avg_speed + 300
      
            dur = random.choice(accomp_note_lengths)
            accomp_attrs[i] = (pitch, vol, dur, pan, lpf)
    

    melody_pitches = []
    for i in range(len(melody_attrs)):
        if melody_attrs[i] is None:
            melody_pitches.append(None)
        else:
            melody_pitches.append(melody_attrs[i][0])
    if len(melody_pitches) == len(prev_melody_pitches):
        melody_pitches = differentiate_pitches(melody_pitches, prev_melody_pitches)
    melody_pitches = refine_pitches(melody_pitches)
    # print("melody:", melody_pitches)

    for i in range(len(melody_attrs)):
        if melody_attrs[i] is None:
            continue
        pitch = melody_pitches[i]
        # pitch = quantize(pitch, [-7,-5,2, 0,2,3,4,5, 7,9,12])
        # print(pitch)
        # pitch = quantize(pitch, chord)
    
        vol = melody_attrs[i][1]
        dur = melody_attrs[i][2]
        pan = melody_attrs[i][3]
        lpf = melody_attrs[i][4]
        delay = random.choice([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
        # print(delay)

        # synth_rand = random.choice(synths)
        players_melody[i] >> synth_dict[melody_synth](pitch, dur=dur, amp=min(1, vol), pan=pan, room=0.5, mix=0.2, sus=1, delay=0, lpf=lpf)
        # print("playing: ", pitches)
    
    for i in range(len(accomp_attrs)):
        if accomp_attrs[i] is None:
            break
        pitch = accomp_attrs[i][0]
        # pitch = quantize(pitch, [-7,-5,2, 0,2,3,4,5, 7,9,12])
        # print(pitch)
        if quantized:
            pitch = quantize(pitch, base_chord)
        # pitch = (pitch, pitch+2, pitch+4)
    
        vol = accomp_attrs[i][1]
        dur = accomp_attrs[i][2]
        pan = accomp_attrs[i][3]
        lpf = accomp_attrs[i][4]
        # delay = random.choice([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
        # print(delay)

        # synth_rand = random.choice(synths)
        players_accomp[i] >> synth_dict[accomp_synth](pitch, dur=dur, amp=min(1, vol), pan=pan, room=0.5, mix=0.2, sus=harmony_sus, delay=0, lpf=lpf)

    prev_melody_pitches = melody_pitches
    return trajectories_best_indices

def show_frame(): 
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

    # vid_frame += 1
    # if vid_frame % chord_change_interval == 0:
    #     chord_idx += 1
    #     if chord_idx == len(chords):
    #         chord_idx = 0
    #     chord = chords[chord_idx]
    #     print("CHORD IS ", chord_idx)
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

    frame = image_resize(frame, width=VIDEO_W)

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

    imgtk = convert_for_tk(last_frame)
    result.imgtk = imgtk
    result.configure(image=imgtk)
    result.after(10, show_frame)



if __name__ == '__main__':
    def pause_playback():
        global playing
        playing = False
        play_btn.config(text="Generate")
        fd.Clock.clear()
        root_dropdown.config(state=tk.NORMAL)
        chords_dropdown.config(state=tk.NORMAL)
        scale_dropdown.config(state=tk.NORMAL)
        melody_synth_dropdown.config(state=tk.NORMAL)
        accomp_synth_dropdown.config(state=tk.NORMAL)

    def start_playback():
        global playing, selected_video, webcam
        playing = True
        play_btn.config(text="Pause")
        root_dropdown.config(state=tk.DISABLED)
        chords_dropdown.config(state=tk.DISABLED)
        scale_dropdown.config(state=tk.DISABLED)
        melody_synth_dropdown.config(state=tk.DISABLED)
        accomp_synth_dropdown.config(state=tk.DISABLED)

        show_frame()

    def stop_playback():
        global cap, cap_exists, selected_video, trajectories, webcam
        play_btn.config(text="Generate")

        if webcam:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(selected_video)

        flag, frame = cap.read()
        if webcam:
            h, w = frame.shape[:2]
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        pause_playback()
        cap.release()
        imgtk = convert_for_tk(frame)
        result.imgtk = imgtk
        result.configure(image=imgtk)
        trajectories = []
        cap_exists = False
        
    root=tk.Tk()                                     
    root.title("Kinetic Soundscapes")            #you can give any title

    sidebar = tk.LabelFrame(root, width=300, height=VIDEO_W, borderwidth=2, padx=5, pady=5, relief='raised')
    sidebar.grid(column=0, sticky='ns')

    input = tk.LabelFrame(sidebar, text="Input", width=300, padx=5, pady=5)
    input.grid(sticky='ew', columnspan=2)
    input.columnconfigure(0, weight = 1)
    input.columnconfigure(1, weight = 1)

     # file dialog
    root.filename = ""
    selected_video = ""
    var = tk.StringVar()

    def select_file():
        global selected_video, webcam
        webcam = False
        pause_playback()
        root.filename = tk.filedialog.askopenfilename(initialdir=getcwd()+'/media', title="Select a video file (mp4)", filetypes=(("mp4 files", "*.mp4"),("all files", "*.*")))
        if (root.filename != "") :
            selected_video = root.filename
        # print("selected_video is ", selected_video)
        var.set(basename(normpath(selected_video)))

        root.focus_force()

        if root.filename != "":
            cap = cv2.VideoCapture(root.filename)
            flag, frame = cap.read()
            frame = image_resize(frame, width=VIDEO_W)
            imgtk = convert_for_tk(frame)
            result.imgtk = imgtk
            result.configure(image=imgtk)
            stop_playback()

    select_file_button = tk.Button(input, text="Select Video File", command=select_file, height=2)
    select_file_button.grid(column=0, row=0, sticky='we')

    selected_file_label = tk.Label(input, textvariable=var, font=("Helvetica Bold", 14))
    selected_file_label.grid(pady=(2,0), columnspan=2, sticky='w')

    def use_webcam():
        global webcam
        webcam = True    
        stop_playback()
    use_webcam_button = tk.Button(input, text="Use Webcam", command=use_webcam, height=2)
    use_webcam_button.grid(column=1, row=0, sticky='we')

    sidebar_motion = tk.LabelFrame(sidebar, text="Motion Settings", width=280, height=360, padx=5, pady=5)
    sidebar_motion.grid(sticky='ew', pady=5, columnspan=2)

    sidebar_motion.columnconfigure(0, weight = 1)

    maxcorners_label = tk.Label(sidebar_motion, text="Max Corners").grid(row=0, column=0, sticky='sw')
    trajlen_label = tk.Label(sidebar_motion, text="Trajectory Length").grid(row=1, column=0,  sticky='sw')
    detectinterval_label = tk.Label(sidebar_motion, text="Detect Interval").grid(row=2, column=0, sticky='sw')

    # slider for max corners
    def slide_maxcorners(var):
        global max_corners, default_maxcorners
        max_corners = maxcorners_slider.get()
    maxcorners_slider = tk.Scale(sidebar_motion, from_=1, to=20, orient=tk.HORIZONTAL, resolution = 1, length = 150, sliderlength=20, command=slide_maxcorners)
    maxcorners_slider.set(default_maxcorners)
    maxcorners_slider.grid(row=0, column=1)

    # slider for trajectory length
    def slide_trajlen(var):
        global trajectory_len, default_trajlen
        trajectory_len = trajlen_slider.get()
    trajlen_slider = tk.Scale(sidebar_motion, from_=2, to=160, orient=tk.HORIZONTAL, resolution = 1, length = 150, sliderlength=20, command=slide_trajlen)
    trajlen_slider.set(default_trajlen)
    trajlen_slider.grid(row=1, column=1)

    # slider for detection interval
    def slide_detectinterval(var):
        global detect_interval, default_detectinterval
        detect_interval = detectinterval_slider.get()
    detectinterval_slider = tk.Scale(sidebar_motion, from_=1, to=8, orient=tk.HORIZONTAL, resolution = 1, length = 150, sliderlength=20, command=slide_detectinterval)
    detectinterval_slider.set(default_detectinterval)
    detectinterval_slider.grid(row=2, column=1)

    def reset_motion_settings():
        global default_maxcorners, default_trajlen, default_detectinterval
        maxcorners_slider.set(default_maxcorners)
        trajlen_slider.set(default_trajlen)
        detectinterval_slider.set(default_detectinterval)
    reset_motion_btn = tk.Button(sidebar_motion, text="Reset", command=reset_motion_settings)
    reset_motion_btn.grid(columnspan=2)

    sidebar_music = tk.LabelFrame(sidebar, text="Music Settings", width=280, height=500, padx=5, pady=5)
    sidebar_music.grid(sticky='ew', pady=5, columnspan=2)
    sidebar_music.columnconfigure(0, weight = 1)

    player = tk.LabelFrame(root, width=1300, height=1000, borderwidth=2, relief='raised')
    player.grid(column=1, row=0, sticky='ns')

    viewer = tk.LabelFrame(player, width=1300, height=900, borderwidth=2, relief='sunken')
    viewer.grid(row=0)

    playbar = tk.LabelFrame(player, width=1300, height=100, borderwidth=0)
    playbar.grid(row=1)

    result = tk.Label(viewer)
    result.grid()

    img = Image.fromarray(np.zeros((480,VIDEO_W,3), np.uint8))
    imgtk = ImageTk.PhotoImage(image=img)
    result.imgtk = imgtk
    result.configure(image=imgtk)

    root_label = tk.Label(sidebar_music, text="Root").grid(row=0, column=0, sticky='ws')
    chords_label = tk.Label(sidebar_music, text="Chords").grid(row=1, column=0, sticky='ws')
    scale_label = tk.Label(sidebar_music, text="Scale").grid(row=2, column=0, sticky='ws')
    tempo_label = tk.Label(sidebar_music, text="Tempo").grid(row=3, column=0, sticky='ws')
    # cci_label = tk.Label(sidebar_music, text="Chord Change Interval").grid(row=3, column=0, sticky='ws')
    melody_synth_label = tk.Label(sidebar_music, text="Melody Synth").grid(row=4, column=0, sticky='ws')
    accomp_synth_label = tk.Label(sidebar_music, text="Harmony Synth").grid(row=5, column=0, sticky='ws')
    melody_layers_label = tk.Label(sidebar_music, text="Melody Layers").grid(row=6, column=0, sticky='ws')
    accomp_layers_label = tk.Label(sidebar_music, text="Harmony Layers").grid(row=7, column=0, sticky='ws')
    harmony_sus_label = tk.Label(sidebar_music, text="Harmony Sustain").grid(row=8, column=0, sticky='ws')


    # dropdown for root
    def set_root(var):
        global base_root
        # fd.Root.default.set(selected_root.get())
        base_root = selected_root.get()
        # print("setting root to ", fd.Root.default.char)
    selected_root = tk.StringVar()
    selected_root.set('C')
    root_dropdown = tk.OptionMenu(sidebar_music, selected_root, 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B', command=set_root)
    root_dropdown.grid(row=0, column=1, sticky='ew')

    # dropdown for chords setting
    def set_chords(var):
        global chord_setting
        chord_setting = selected_chords.get()
    selected_chords = tk.StringVar()
    selected_chords.set(default_chord_setting)
    chords_dropdown = tk.OptionMenu(sidebar_music, selected_chords, "none", "cinematic I VI III IV", "basic I IV V I", command=set_chords)
    chords_dropdown.grid(row=1, column=1, sticky='ew')


    # dropdown for scale
    def set_scale(var):
        global quantized
        if selected_scale.get() == 'none (atonal)':
            quantized = False
        else:
            quantized = True
            fd.Scale.default.set(selected_scale.get())
        print("setting scale to ", fd.Scale.default.name)
    selected_scale = tk.StringVar()
    selected_scale.set('major')
    scale_dropdown = tk.OptionMenu(sidebar_music, selected_scale, 'major', 'minor', 'none (atonal)', 'aeolian', 'altered', 'bebopDom', 'bebopDorian', 'bebopMaj', 'bebopMelMin', 'blues', 'chinese', 'chromatic', 'custom', 'default', 'diminished', 'dorian', 'dorian2', 'egyptian', 'freq', 'halfDim', 'halfWhole', 'harmonicMajor', 'harmonicMinor', 'hungarianMinor', 'indian', 'justMajor', 'justMinor', 'locrian', 'locrianMajor', 'lydian', 'lydianAug', 'lydianDom', 'lydianMinor', 'majorPentatonic', 'melMin5th', 'melodicMajor', 'melodicMinor', 'minMaj', 'minorPentatonic', 'mixolydian', 'phrygian', 'prometheus', 'romanianMinor', 'susb9', 'wholeHalf', 'wholeTone', 'yu', 'zhi', command=set_scale)
    scale_dropdown['menu'].insert_separator(3)
    scale_dropdown.grid(row=2, column=1, sticky='ew')

    # slider for tempo
    def slide_bpm(var):
        fd.Clock.update_tempo_now(bpm_slider.get())
    bpm_slider = tk.Scale(sidebar_music, from_=20, to=220, orient=tk.HORIZONTAL, resolution = 4, length = 150, sliderlength=20, command=slide_bpm)
    bpm_slider.set(120)
    bpm_slider.grid(row=3, column=1)

    # # slider for chord change interval
    # def slide_cci(var):
    #     global chord_change_interval
    #     chord_change_interval = cci_slider.get()
    # cci_slider = tk.Scale(sidebar_music, from_=10, to=500, orient=tk.HORIZONTAL, resolution = 5, length = 150, sliderlength=20, command=slide_cci)
    # cci_slider.set(default_cci)
    # cci_slider.grid(row=3, column=1)

    # dropdown for melody synth
    def set_melody_synth(var):
        global melody_synth
        melody_synth = selected_melody_synth.get()
    selected_melody_synth = tk.StringVar()
    selected_melody_synth.set(default_melody_synth)
    melody_synth = default_melody_synth
    
    # MELODIC SYNTHS
    # percussive: marimba, donk, space, bell, gong, piano, keys
    # plucked: karp, pluck, sitar
    # gentle: sinepad, blip
    # bright: nylon, scatter, charm
    melody_synth_options = ['piano',  'sinepad', 'nylon', 'scatter', 'charm',  'karp', 'pluck', 'sitar',  'marimba', 'donk', 'space', 'bell', 'gong', 'blip', 'keys', 'soprano']
    melody_synth_dropdown = tk.OptionMenu(sidebar_music, selected_melody_synth, *melody_synth_options, command=set_melody_synth)
    melody_synth_dropdown.grid(row=4, column=1, sticky='ew')
    melody_synth_dropdown['menu'].insert_separator(2)
    melody_synth_dropdown['menu'].insert_separator(5)
    melody_synth_dropdown['menu'].insert_separator(8)

    # dropdown for accomp synth
    def set_accomp_synth(var):
        global accomp_synth
        accomp_synth = selected_accomp_synth.get()
    selected_accomp_synth = tk.StringVar()
    selected_accomp_synth.set(default_accomp_synth)
    accomp_synth = default_accomp_synth

    # ACCOMP SYNTHS
    # gentle:  sinepad, soft, blip, keys, zap
    # ambient: ambi, klank, glass, space, soprano,
    # bright: nylon, pulse, scatter, charm, ripple, creep, bug ( no melod), saw, pads
    # dark: dub, bass, jbass, varsaw, lazer
    # percussive: bell, gong, pluck, piano
    accomp_synth_options = ['piano', 'ambi', 'klank', 'glass', 'space', 'soprano',  'nylon', 'pulse', 'scatter', 'charm', 'ripple', 'creep', 'bug', 'saw', 'pads',  'dub', 'bass', 'jbass', 'varsaw', 'lazer',  'sinepad', 'soft', 'blip', 'keys', 'zap',  'bell', 'gong', 'pluck',]
    accomp_synth_dropdown = tk.OptionMenu(sidebar_music, selected_accomp_synth, *accomp_synth_options, command=set_accomp_synth)
    accomp_synth_dropdown.grid(row=5, column=1, sticky='ew')
    accomp_synth_dropdown['menu'].insert_separator(6)
    accomp_synth_dropdown['menu'].insert_separator(16)
    accomp_synth_dropdown['menu'].insert_separator(22)
    accomp_synth_dropdown['menu'].insert_separator(28)

    # slider for melody layers
    def slide_melody_layers(var):
        global melody_layers
        melody_layers = melody_layers_slider.get()
    melody_layers_slider = tk.Scale(sidebar_music, from_=0, to=8, orient=tk.HORIZONTAL, resolution = 1, length = 150, sliderlength=20, command=slide_melody_layers)
    melody_layers_slider.set(default_melody_layers)
    melody_layers_slider.grid(row=6, column=1)

    # slider for accomp layers
    def slide_accomp_layers(var):
        global accomp_layers
        accomp_layers = accomp_layers_slider.get()
    accomp_layers_slider = tk.Scale(sidebar_music, from_=0, to=8, orient=tk.HORIZONTAL, resolution = 1, length = 150, sliderlength=20, command=slide_accomp_layers)
    accomp_layers_slider.set(default_accomp_layers)
    accomp_layers_slider.grid(row=7, column=1)

    # slider for harmony sustain level
    def slide_harmony_sus(var):
        global harmony_sus
        harmony_sus = harmony_sus_slider.get()
    harmony_sus_slider = tk.Scale(sidebar_music, from_=0, to=10, orient=tk.HORIZONTAL, resolution = 1, length = 150, sliderlength=20, command=slide_harmony_sus)
    harmony_sus_slider.set(default_harmony_sus)
    harmony_sus_slider.grid(row=8, column=1)


    toggles = tk.LabelFrame(playbar, borderwidth=0)
    toggles.grid(column=0)
    # toggle video on/off
    def toggle_video():
        global show_video
        show_video = video_toggle_var.get() == 1
    video_toggle_var = tk.IntVar(value=1)
    video_toggle = tk.Checkbutton(toggles, text="Show Video", variable=video_toggle_var, command=toggle_video)
    video_toggle.grid(column=0, row=0, sticky='w')

    # toggle flow on/off
    def toggle_flow():
        global show_flow
        show_flow = flow_toggle_var.get() == 1
    flow_toggle_var = tk.IntVar(value=1)
    flow_toggle = tk.Checkbutton(toggles, text="Show Flow", variable=flow_toggle_var, command=toggle_flow)
    flow_toggle.grid(column=1, row=0, sticky='w')

    playstop = tk.LabelFrame(playbar, borderwidth=0)
    playstop.grid()
    # play/pause button
    def switch():
        global playing
        if playing:
            pause_playback()
        else:
            if selected_video != "" or webcam == True:
                start_playback()

    play_btn = tk.Button(playstop, text="Generate", command=switch, height=3, width=6, relief='raised')
    play_btn.grid(column=0, row=0, pady=5)


    # stop button
    stop_btn = tk.Button(playstop, text="Stop", command=stop_playback, height=3, width=6, relief='raised')
    stop_btn.grid(column=1, row=0, pady=5)


    
    if selected_video != "" or webcam == True:
        show_frame()

    root.mainloop()                                  #keeps the application in an infinite loop so it works continuosly
