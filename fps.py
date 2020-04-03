from grabscreen import *
import numpy as np
import cv2
import pyautogui
from directkeys import PressKey, ReleaseKey, W, A, S, D
from mouse import *
from objectdetection import *
import sys
import time
import imutils
from cv2detectionfromgraph import CVTFObjectDetection

def process_img(img):
    # convert to gray scale
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img

def fps_counter(fps):
    t = time.time()
    sys.stdout.write("FPS  %f   \r" % (1 / (t-fps)))
    return time


def get_selected_img(x1, y1, x2, y2):

    # I calculated this ratio, as it is 16:9 and contains the most focal point of the game
    xdiff = 3 * (x2 - x1) / 8
    ydiff = 3 * (y2 - y1) / 8

    x1offset = xdiff
    x2offset = -xdiff
    y1offset = ydiff
    y2offset = -ydiff
    x1 += x1offset
    x2 += x2offset
    y1 += y1offset
    y2 += y2offset

    # put in list and cast to integer pixel position
    region = [int(x1), int(y1), int(x2), int(y2)]

    # grab screen
    img = grab_screen(region=region)
    
    # scale
    #img = imutils.resize(img, width=int((x2-x1) * scale_factor))
    return img

def convert_pixel_locations(x1, y1, x2, y2, box):
    # x1,y1,x2,y2 are the ORIGINAL location of the screen grab
    # box represents coordinates of a detection
    return

def main():
    cv2.setNumThreads(8)
    
    # define region
    x1 = 0
    y1 = 0
    x2 = 1920
    y2 = 1080

    detection = CVTFObjectDetection()

    threshold = 0.7


    while(True):
        t = time.time()

        img = get_selected_img(x1,y1,x2,y2)
        #boxes, scores, classes, num = odapi.processFrame(img)
        img, status = detection.detect(img)
        if status == 1:
            full_auto(1)

        cv2.imshow('detection', img)

        t = fps_counter(t)

        # break loop
        if (cv2.waitKey(25) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
        

if __name__=='__main__':
    main()