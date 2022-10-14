import numpy
import cv2
import time
from FoxDot import *
import math
import os


print(SynthDefs)

def generate_sound(flow_window, step=16):
    Clock.bpm = 108
    h, w = img.shape[:2]
    y, x = numpy.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)

    Mag = []
    Dir = []
    for flow in flow_window:
        fx, fy = flow[y,x].T

        num_rows = int(h/step)
        num_cols = int(w/step)

        split_fx = numpy.split(fx, num_rows)
        split_fy = numpy.split(fy, num_rows)

        # absolute value of flow
        fx_abs = [abs(val) for val in fx]
        fy_abs = [abs(val) for val in fy]

        # magnitude of flow
        magnitudes = []
        for valx in fx:
            for valy in fy:
                mag = math.sqrt(valx**2 + valy**2)
                magnitudes.append(mag)
        avg_magnitude = sum(magnitudes) / len(magnitudes)
        Mag.append(avg_magnitude)

        # direction of flow
        X = 0
        Y = 0
        for valx in fx:
            X += valx
        for valy in fy:
            Y += valy

        # print("X: ", X, " Y: ", Y)
        avg_dir = math.atan2(Y, X) + math.pi

        Dir.append(avg_dir)

        # p5 >> pluck(round(avg_dir))


        # average magnitude of flow (horz, vert)
        avg_fx_abs = sum(fx_abs)/len(fx)
        avg_fy_abs = sum(fy_abs)/len(fy)
        
        left_flow = 0
        right_flow = 0
        row_weights = [0] * num_rows

        for row in range(len(split_fx)):
            for column in range(num_cols//2):
                flow = abs(split_fx[row][column]) + abs(split_fy[row][column])
                left_flow += flow
                row_weights[row] += flow
            for column in range(num_cols//2+1, num_cols):
                flow = abs(split_fx[row][column]) + abs(split_fy[row][column])
                right_flow += flow
                row_weights[row] += flow
        row_weights.reverse()
        row_weight_largest = max(row_weights)
        dominant_row = row_weights.index(row_weight_largest)

    # Mag = sum(Mag)/len(Mag)
    # # print("magnitude of window: ", Mag)
    # Dir = sum(Dir)/len(Dir)
    # if Mag > 0:
    #     p1 >> piano(Dir, dur=1/2, amp=min(3, Mag))

    #     print("dir of window: ", Dir)
    # else:
    #     Clock.clear()




    # p1 >> piano(1, dur=1/2, amp=min(0.2*Mag, 3))
    
    tone1 = (5, 8)
    bass1 = 3.5
    
    intensity1 = 0.5*left_flow/(num_rows * num_cols//2)
    intensity2 = 0.5*right_flow/(num_rows * num_cols//2)
    intensity3 = (intensity1 + intensity2) / 2
    vol1 = 0
    vol2 = 0
    vol3 = 0
    if intensity1 > 0.3:
        vol1 = 0.1*intensity1
    if intensity2 > 0.3:
        vol2 = 0.2*intensity2
    if intensity3 > 0.3:
        vol3 = 0.2*intensity3
        
    max_vol = 5
    p1 >> piano(tone1, dur=1/4, amp=min(vol1, max_vol))
    p2 >> klank(bass1, dur=1/4, amp=min(vol2, max_vol))

    tone2 = tone1[1]
    bass2 = bass1
    if dominant_row == 1:
        tone2 += 1
        bass2 += 2.5
    if dominant_row == 2:
        tone2 = tone1[1] + 4
        bass2 += 2.5
    p3 >> piano(tone2, dur=1/4, amp=min(vol1, max_vol))
    p4 >> klank(bass2, dur=1/4, amp=min(vol2, max_vol))


    # fx, fy = flow[:,:,0], flow[:,:,1]

    # ang = numpy.arctan2(fy, fx) + numpy.pi
    # print('ang is ', ang)
    # avg_ang = sum(ang)/len(ang)
    # print("AVG ANG IS ", avg_ang)
    

def draw_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = numpy.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = numpy.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = numpy.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # img_bgr = numpy.zeros((h, w, 3), numpy.uint8)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


def draw_hsv(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = numpy.arctan2(fy, fx) + numpy.pi

    v = numpy.sqrt(fx*fx+fy*fy)



    hsv = numpy.zeros((h, w, 3), numpy.uint8)
    hsv[...,0] = ang*(180/numpy.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = numpy.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr




cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("video_clips/waves1.mp4")


suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:

    motion_window = 1
    flow_window = []

    for i in range(motion_window):

        suc, img = cap.read()
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        # cv2.imwrite(os.path.join(os.getcwd(), 'captures' + str(j) + ' ' + str(i) + '.png'), img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(gray,2,0.05,5)
        corners = numpy.int0(corners)

        # blank = numpy.zeros((h, w, 3), numpy.uint8)
        for i in corners:
            x,y = i.ravel()
            cv2.circle(img,(x,y),3,255,-1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print("LEN CORNERS ", len(corners))

        # start time to calculate FPS
        start = time.time()


        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.1, 2, 10, 2, 1, 1, 0)
        flow_window.append(flow)
        prevgray = gray


        # End time
        end = time.time()
        # calculate the FPS for current frame detection
        fps = 1 / (end-start)

        print(f"{fps:.2f} FPS")

        cv2.imshow('flow', draw_flow(gray, flow))
        cv2.imshow('flow HSV', draw_hsv(flow))
        key = cv2.waitKey(5)
        if key == ord('q'):
            break
        
    # generate_sound(flow_window)

    # suc, img = cap.read()
    # img = cv2.flip(img, 1)
    # h, w = img.shape[:2]
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # corners = cv2.goodFeaturesToTrack(gray,250,0.01,10)
    # corners = numpy.int0(corners)
    # # print("CORNERS ", corners)

    # blank = numpy.zeros((h, w, 3), numpy.uint8)
    # for i in corners:
    #     x,y = i.ravel()
    #     cv2.circle(blank,(x,y),3,255,-1)
    # cv2.imshow('corners', blank)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break



Clock.clear()
cap.release()
cv2.destroyAllWindows()
