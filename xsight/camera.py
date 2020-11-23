import cv2
import glob
import numpy as np
import os
from detection import *


def take_photo(filename):
    ret, frame = cap.read()
    cv2.imwrite(filename, frame)

def record_video(filename, mode = 'r'):
    i = 0
    while(True):
        ret, frame = cap.read()
        cv2.imshow('video', frame)
        if( mode == 'r'):
            cv2.imwrite(filename + str(i) + '.jpg', frame)
            i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def real_time_detection():
    d = Detection()

    i = 0
    while(True):
        ret, frame = cap.read()
        frame = d.object_detection(frame)

        cv2.imshow('video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def write_to_video(dir, videoName):
    img_arr = []
    for filename in sorted(glob.glob(dir + '/*.jpg'), key = os.path.getmtime):
        print(filename)
        img = cv2.imread(filename)
        img_arr.append(img)

    height, width, layers = img_arr[0].shape
    size = (width, height)
    out = cv2.VideoWriter(videoName + '.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    for i in range(len(img_arr)):
        out.write(img_arr[i])

    out.release()


if __name__=="__main__":
    # Specify device
    cap = cv2.VideoCapture(0)
    real_time_detection()
