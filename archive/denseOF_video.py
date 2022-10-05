import numpy as np
import cv2 as cv
import os

cap = cv.VideoCapture("musicvid.mp4")
result = cap.read()
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

i=1
while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 2, 0.2, 0)
    # cv.calcOpticalFlowFarneback(	prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags	)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)  

    cv.imwrite(os.path.join(os.getcwd(), 'captures', 'opticalfb', 'opticalfb-' + str(i) + '.png'), frame2)
    cv.imwrite(os.path.join(os.getcwd(), 'captures', 'opticalhsv', 'opticalhsv-' + str(i) + '.png'), bgr)

    k = cv.waitKey(30) & 0xff
    if k == ord('q'):
        break
    # elif k == ord('s'):
    #     cv.imwrite('opticalfb.png', frame2)
    #     cv.imwrite('opticalhsv.png', bgr)
    prvs = next
    i+=1
cv.destroyAllWindows()