import numpy as np
import cv2
from mouse import *
import sys
import time
from vision import Vision


def main():
    # x1, y1, x2, y2
    window = [0, 0, 1920, 1080]
    # n defines the height and width of our square viewing region
    n = 450
    roi = get_roi(window, n)

    printFps = True
    printStatus = False
    

    vision = Vision(window, roi, n)
    while(True):
        img =  vision.update_frame()
        vision.single_detection()
        cv2.imshow('detection', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def get_roi(window, n):
    n = n/2
    # returns n by n region of interest in the center of the screen
    xdiff = window[2] - window[0]
    ydiff = window[3] - window[1]
    xcenter = xdiff / 2
    ycenter = ydiff / 2
    roi =  [int(xcenter - n), int(ycenter - n), int(xcenter + n), int(ycenter + n)]
    return roi


def get_coords(window, objectRegion, n):
    n = n/2
    # returns the real location of the boxes relative to the gameplay screen
    x1 = window[0]
    y1 = window[1]
    x2 = window[2]
    y2 = window[3]

    roixorgin = int(((x2 - x1) / 2) - n)
    roiyorgin = int(((y2 - y1) / 2) - n)

    objectRegion = [objectRegion[0] + roixorgin,
                    objectRegion[1] + roiyorgin,
                    objectRegion[2] + roixorgin,
                    objectRegion[3] + roiyorgin]

    return objectRegion

if __name__ == '__main__':
    main()