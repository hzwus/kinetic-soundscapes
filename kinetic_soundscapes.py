import numpy as np
import cv2
import time



def draw_flow(img, flow, step=32):
    print("screen dimensions (h, w): ", img.shape[:2])

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    print("shape is ", np.shape(flow[y,x]))
    print("flow is:\n", flow[y,x].T)

    num_rows = int(h/step)
    print("num rows: ", num_rows)
    num_cols = int(w/step)

    split_fx = np.split(fx, num_rows)
    split_fy = np.split(fy, num_rows)

    # absolute value of flow
    fx_abs = [abs(val) for val in fx]
    fy_abs = [abs(val) for val in fy]

    # average magnitude of flow (horz, vert)
    avg_fx_abs = sum(fx_abs)/len(fx)
    avg_fy_abs = sum(fy_abs)/len(fy)
    print("avg_flow_magnitude: ", (avg_fx_abs + avg_fy_abs) / 2)

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
    row_weight_largest = max(row_weights)
    print("dominant row is ", row_weights.index(row_weight_largest))

    intensity1 = left_flow/(num_rows * num_cols//2)
    intensity2 = right_flow/(num_rows * num_cols//2)
    print("intensity1: ", intensity1)
    print("intensity2: ", intensity2)
    


    # # total flow magnitude of left half of screen
    # left_flow = np.sum(fx_abs[:1800]) + np.sum(fy_abs[:1800])

    # right_flow = np.sum(fx_abs[1800:]) + np.sum(fy_abs[1800:])

    print("left: ", left_flow, "right: ", right_flow)

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


def draw_hsv(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    # print("ANGLE IS ", ang)
    v = np.sqrt(fx*fx+fy*fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr




cap = cv2.VideoCapture(0)

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)


while True:

    suc, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    start = time.time()


    # flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.1, 2, 10, 2, 1, 1, 0)

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


cap.release()
cv2.destroyAllWindows()