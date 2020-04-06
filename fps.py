from grabscreen import *
import numpy as np
import cv2
import pyautogui
from directkeys import PressKey, ReleaseKey, W, A, S, D
from mouse import *
import sys
import time
import imutils
from cv2detection import ObjectDetection, Tracker


def fps_counter(fps, print):
    t = time.time()
    fps = 1 / (t - fps)
    if print:
        sys.stdout.write("FPS  %.2f \r" % (fps) )
    return t

def status_update(text, print):
    if print == True:
        sys.stdout.write("\t\t  %s \r" % (text) )


def main():
    # x1, y1, x2, y2
    window = [0, 0, 1920, 1080]
    
    # n defines the height and width of our square viewing region
    n = 300
    # view that the detection algo sees
    roi = get_roi(window, n)
    print(roi)

    detection = ObjectDetection()
    threshold = 0.7
    
    printFps = True
    printStatus = False

    while(True):
        t = time.time()

        img = grab_screen(region=roi)
        img, status, box = detection.detect(img)
        t = fps_counter(t, printFps)
        status_update("No Target", printStatus)

        # taget acquired, begin tracking
        if status == True:
            objectRegion = get_coords(window, box, n)
            tracker = Tracker(window, box, objectRegion, img, n)
            tracker.lock_on()
            # lock onto target
            #lock_on(img, box, x1, y1, x2, y2, n) 
            #status_update('Target Acquired', printStatus)
            

       # cv2.imshow('target', img)
        # break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()