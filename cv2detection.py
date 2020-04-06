import cv2
import sys
from grabscreen import *

classNames = {0: 'background',
                    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
                    7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
                    13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
                    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
                    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
                    32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
                    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
                    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
                    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
                    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
                    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
                    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
                    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

class Tracker():
    def __init__(self, window, roi, box, img, n):
        self.window = window
        self.box = box
        self.roi = roi
        self.n = n
        self.x1 = roi[0]
        self.y1 = roi[1]
        self.x2 = roi[2]
        self.y2 = roi[3]
        self.box = tuple(box)
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(img, self.box)
        print('Tracker initialized')

    def lock_on(self):
        lockedOn = True
        i = 0
        # basically an enhanced detection period
        while lockedOn == True:
            img = grab_screen(region=self.roi)
            # Update tracker
            lockedOn, box = self.tracker.update(img)
            i+=1
            print(i)
            # Draw bounding box
            p1 = (int(self.box[0]), int(self.box[1]))
            p2 = (int(self.box[0] + self.box[2]), int(self.box[1] + self.box[3]))

            # get coordinates relative to game not roi
            targetBox = get_coords(self.roi, box, self.n)
            xcenter = ((targetBox[2] - targetBox[0]) / 2) + targetBox[0]
            ycenter = ((targetBox[3] - targetBox[1]) / 2) + targetBox[1]
                
            # fire on target

            cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
            cv2.imshow('target', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


class ObjectDetection():
    def __init__(self, num_workers=8):
        cv2.setNumThreads(num_workers)
        self.confidence = 0.8
        # Pretrained classes in the model
        self.model = model = cv2.dnn.readNetFromTensorflow('models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb',
                                                          'models/ssd_mobilenet_v2_coco_2018_03_29/config.pbtxt')
        
    def id_class_name(self, class_id, classes):
        for key, value in classes.items():
            if class_id == key:
                return value

    def detect(self, img):
        # Loading model
    
        img_height, img_width, _ = img.shape
        self.model.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True))
        output = self.model.forward()
        
        # keeps track of if there is a person or not
        status = False
        # the box around the target
        # initially -1
        objectRegion = [-1, -1, -1, -1]

        for detection in output[0, 0, :, :]:
            class_id = detection[1]
            confidence = detection[2]
            if class_id == 1 and detection[2] > self.confidence:
                class_name=self.id_class_name(class_id,classNames)
                box_x = detection[3] * img_width
                box_y = detection[4] * img_height
                box_width = detection[5] * img_width
                box_height = detection[6] * img_height
                cv2.rectangle(img, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
                cv2.putText(img,class_name ,(int(box_x), int(box_y+.05*img_height)),cv2.FONT_HERSHEY_SIMPLEX,(.005*img_width),(0, 0, 255))
                objectRegion = [int(box_x), int(box_y), int(box_x + box_width), int(box_y + box_height)]
                status = True
                return img, status, objectRegion

        return img, status, objectRegion
    

