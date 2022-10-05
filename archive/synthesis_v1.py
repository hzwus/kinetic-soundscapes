from fileinput import filelineno
import numpy
import cv2
import time
import math
from FoxDot import *
import itertools
import random
from functools import cmp_to_key
from util import quantize
# import supriya

# server = supriya.Server().boot()

# NOTE: VARIABLE NAMES WITH 2 LETTERS ARE ONLY TO BE USED FOR FOXDOT PLAYER OBJECTS. (THEY ARE RESERVED IN THE FOXDOT NAMESPACE)

# generate all permutations of 2 letter words
# these are our FoxDot player objects
# letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# players = [''.join(i) for i in itertools.product(letters, repeat = 2)]
random.seed(time.time())
players = [a1, a2, a3, a4, a5, a6, a7, a8]
# players.extend([b1, b2, b3, b4])
synths = [marimba, ambi, sinepad]
# synths = [klank, feel, ambi]

chords = [
            [-7,-5,2, 0,2,3,4,5, 7,9,12], 
            [-10,-8,-6, -3,-1,1,2,3, 4,6,8,9],
            [-11, -9, -7, -4,-2,0, 3, 5, 7]
            ]
max_players = len(players)

# ['aeolian', 'altered', 'bebopDom', 'bebopDorian', 'bebopMaj', 'bebopMelMin', 'blues', 'chinese', 'chromatic', 'custom', 'default', 'diminished', 'dorian', 'dorian2', 'egyptian', 'freq', 'halfDim', 'halfWhole', 'harmonicMajor', 'harmonicMinor', 'hungarianMinor', 'indian', 'justMajor', 'justMinor', 'locrian', 'locrianMajor', 'lydian', 'lydianAug', 'lydianDom', 'lydianMinor', 'major', 'majorPentatonic', 'melMin5th', 'melodicMajor', 'melodicMinor', 'minMaj', 'minor', 'minorPentatonic', 'mixolydian', 'phrygian', 'prometheus', 'romanianMinor', 'susb9', 'wholeHalf', 'wholeTone', 'yu', 'zhi']

Scale.default = "major"
Root.default.set("C#")

lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 20,
                    qualityLevel = 0.3,
                    minDistance = 10,
                    blockSize = 7 )


trajectory_len = 16
detect_interval = 1
trajectories = []
frame_idx = 0


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("media/fireworks.mp4")


# synth = server.add_synth()

chord = chords[0]
chord_change_interval = 140
vid_frame = 0
chord_idx = 0
while True:

    vid_frame += 1
    # print(vid_frame)
    if vid_frame % chord_change_interval == 0:
        chord_idx += 1
        if chord_idx == len(chords):
            chord_idx = 0
        chord = chords[chord_idx]
        print("CHORD IS ", chord_idx)



    # start time to calculate FPS
    start = time.time()

    suc, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.flip(frame_gray, 1)
    img = frame.copy()
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

        # mag_total = 0
        # for t in trajectories:
        #     mag = math.dist(t[0], t[-1])
        #     mag_total += mag
        # print("mag total is ", mag_total)

        # vol = mag_total / 100
        # pitches = set()
        player_attrs = [None] * len(players)
        
            
        # print(len(trajectories))
        if len(trajectories) > max_players:
            trajectories_sorted = sorted(trajectories, key=cmp_to_key(lambda t1, t2: math.dist(t2[0], t2[-1]) - math.dist(t1[0], t1[-1])))

            trajectories_best = trajectories_sorted[:max_players]
            # for t1 in trajectories_best:
            #     print(math.dist(t1[0], t1[-1]))
            # print("DONE")
        else:
            trajectories_best = trajectories
        for i in range(len(trajectories_best)):
            t = trajectories_best[i]
            mag = math.dist(t[0], t[-1])
            vol = mag / 100
            pitch = round((h - t[-1][1]) / h * 24 - 12)
            dur = 1/3
            pan = (t[-1][0] / w) * 1.6 - 0.8
            player_attrs[i] = (pitch, vol, dur, pan)

            # print(mag)
        # print(player_attrs)
        
        # ['loop', 'stretch', 'play1', 'play2', 'audioin', 'noise', 'dab', 'varsaw', 'lazer', 'growl', 'bass', 'dirt', 'crunch', 'rave', 'scatter', 'charm', 'bell', 'gong', 'soprano', 'dub', 'viola', 'scratch', 'klank', 'feel', 'glass', 'soft', 'quin', 'pluck', 'spark', 'blip', 'ripple', 'creep', 'orient', 'zap', 'marimba', 'fuzz', 'bug', 'pulse', 'saw', 'snick', 'twang', 'karp', 'arpy', 'nylon', 'donk', 'squish', 'swell', 'razz', 'sitar', 'star', 'jbass', 'piano', 'sawbass', 'prophet', 'pads', 'pasha', 'ambi', 'space', 'keys', 'dbass', 'sinepad']
        for i in range(len(player_attrs)):
            if player_attrs[i] is None:
                break
            pitch = player_attrs[i][0]
            # pitch = quantize(pitch, [-7,-5,2, 0,2,3,4,5, 7,9,12])
            pitch = quantize(pitch, chord)

            vol = player_attrs[i][1]
            dur = player_attrs[i][2]
            pan = player_attrs[i][3]

            synth_rand = random.choice(synths)
            
            players[i] >> synth_rand(pitch, dur=1/4, amp=min(2, vol), pan=pan, room=0.5, mix=0.2, sus=1, delay=0)

      

        # if 'p1' in player_dict:
        #     fd.p1 >> fd.piano(player_dict['p1'][0], dur=1/2, amp=min(2, player_dict['p1'][1]))

        # if 'p2' in player_dict:
        #     fd.p2 >> fd.piano(player_dict['p2'][0], dur=1/2, amp=min(2, player_dict['p2'][1]))
        # if 'p3' in player_dict:
        #     fd.p3 >> fd.piano(player_dict['p3'][0], dur=1/2, amp=min(2, player_dict['p3'][1]))

        # if 'p4' in player_dict:
        #     fd.p4 >> fd.piano(player_dict['p4'][0], dur=1/2, amp=min(2, player_dict['p4'][1]))


        # # print(str(pitches))
        # locals().update(player_dict)
        # for player in player_dict:
        #     pitch, vol = player_dict[player]
        #     fd.player >> fd.piano(pitch, dur=1/2, amp=min(2, vol))




        # Draw all the trajectories
        cv2.polylines(img, [numpy.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
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
    cv2.imshow('Optical Flow', img)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

Clock.clear()
# synth.release()
# server.quit()
cap.release()
cv2.destroyAllWindows()