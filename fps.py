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


def fps_counter(fps, print):
    t = time.time()
    fps = 1 / (t - fps)
    if print:
        sys.stdout.write("FPS  %.2f \r" % (fps) )
    return t

def status_update(text, print):
    sys.stdout.write("\t\t  %s \r" % (text) )


def main():
    # define region
    x1 = 0
    y1 = 0
    x2 = 1920
    y2 = 1080

    detection = CVObjectDetection()

   
    threshold = 0.7
    n = 150
    
    printFps = True
    printStatus = True

    while(True):
        t = time.time()
        img = roi(x1,y1,x2,y2, n=n)
        img, status, box = detection.detect(img)
        t = fps_counter(t, printFps)
        
        status_update("No Target", printStatus)

        # taget acquired, begin tracking
        if status == 1:
            #objectRegion = get_coords(x1, y1, x2, y2, box, n)

            # lock onto target
            lock_on(img, box, x1, y1, x2, y2, n) 
            status_update('Target Lost', True)
        else:
            cv2.imshow('detection', img)
       
        # break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def lock_on(img, box, x1, y1, x2, y2, n):

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
    tracker_type = tracker_types[2]
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()

    # Initialize tracker with first frame and bounding box
    lockedOn = tracker.init(img, tuple(box))
    
    img = roi(x1,y1,x2,y2, n=n)

    # basically an enhanced detection period
    while lockedOn:
        t = time.time()

        # Read a new frame
        img = roi(x1,y1,x2,y2, n=n)
        # Update tracker
        lockedOn, box = tracker.update(img)

        # Draw bounding box
        if lockedOn:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))

            # get coordinates relative to game not roi
            targetBox = get_coords(x1, y1, x2, y2, box, n=n)
            xcenter = ((targetBox[2] - targetBox[0]) / 2) + targetBox[0]
            ycenter = ((targetBox[3] - targetBox[1]) / 2) + targetBox[1]
            
            # fire on target
            #full_auto(.2, xcenter, ycenter)
            quickscope(xcenter, ycenter)
            #aim(1, xcenter, ycenter)

            cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
            cv2.imshow('target', img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            t = fps_counter(t, False)
            status_update('Locked On', True)
        else:

            return

if __name__ == '__main__':
    main()