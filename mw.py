from grabscreen import *
import numpy as np
import cv2
import pyautogui
from directkeys import PressKey, ReleaseKey, W, A, S, D
from mouse import click, get_position, quickscope, gratata
from objectdetection import *
import sys
import time
import imutils

def process_img(img):
    # convert to gray scale
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img





def move_cursor(pos, x1, y1, x2, y2, xoffset, yoffset, crop_factor, scale_factor):
    # undo transformations
    #xdiff = x2 - x1
    #ydiff = y2 - y1

    # undo scale 
    #xscale = (x2 - x1) * (1 / scale_factor)
   # yscale = (y2 - y1) * (1 / scale_factor)
    #pos[0] -= xscale
    #pos[1] -= yscale
    #pos[2] += xscale
    #pos[3] += yscale

    # undo crop
    #xdiff = pos[2] - pos[0]
    #ydiff = pos[3] - pos[1]

    #pos[0] = pos[0] - xdiff * 1/crop_factor
    #pos[1] = pos[1] - ydiff * 1/crop_factor
    #pos[2] = pos[2] + xdiff * 1/crop_factor
    #pos[3] = pos[3] + xdiff * 1/crop_factor

    #pos[0] -= xoffset
    #pos[1] -= yoffset
   # pos[2] += xoffset
    #pos[3] += yoffset
    return pos

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

    model_path = "models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
    #model_path = '/models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_model=model_path)
    threshold = 0.7


    while(True):
        t = time.time()

        img = get_selected_img(x1,y1,x2,y2)
        boxes, scores, classes, num = odapi.processFrame(img)
        for i in range(len(boxes)):
            # shoot if there's a person
            pos = get_position()
            #sys.stdout.write("%f  %f \r" % pos[0], pos[1])

            
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                quickscope()
                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)

        cv2.imshow('capture', img)
        t = fps_counter(t)

        # break loop
        if (cv2.waitKey(25) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
        

if __name__=='__main__':
    main()



if __name__ == "__main__":

    cap = cv2.VideoCapture('/path/to/input/video')

    while True:
        r, img = cap.read()
        img = cv2.resize(img, (1280, 720))

        

        # Visualization of the results of a detection.

        

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break