from grabscreen import *
import numpy as np
import cv2
import pyautogui
from directkeys import PressKey, ReleaseKey, W, A, S, D
from mouse import *
import sys
import time
import imutils
from cv2detection import CVObjectDetection



def fps_counter(fps):
    t = time.time()
    sys.stdout.write("FPS  %f   \r" % (1 / (t-fps)))
    # returns time and fps
    return time

def main():
    # define region
    x1 = 0
    y1 = 0
    x2 = 1920
    y2 = 1080

    detection = CVObjectDetection()

    threshold = 0.7
    n = 150
    while(True):
        t = time.time()

        img = roi(x1,y1,x2,y2, n=n)
        #boxes, scores, classes, num = odapi.processFrame(img)
        img, status, objectRegion = detection.detect(img)

        if status == 1:
            print('target acquired')
            objectRegion = get_coords(x1, y1, x2, y2, objectRegion, n)
            lock_on_target(objectRegion)
            quickscope()

        cv2.imshow('detection', img)

        t = fps_counter(t)

        # break loop
        if (cv2.waitKey(25) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
        

if __name__=='__main__':
    main()