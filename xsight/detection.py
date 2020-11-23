import torchvision
import torchvision.transforms as T
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import threading
from multiprocessing import Pool

# define the objects that we can detect
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]



class Detection():
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def get_prediction(self, img, threshold):
        transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
        img = transform(img) # Apply the transform to the images
        pred = self.model([img]) # Pass the image to the model
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().numpy())

        # make predictions on the image
        # this line of code is a bitch. Need to fix
        try:
            pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
        except:
            return None, None

        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        return pred_boxes, pred_class

    def object_detection(self, img, threshold=0.5, rect_th=1, text_size=1, text_th=1):
        boxes, pred_cls = self.get_prediction(img, threshold) # Get predictions
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
        if boxes != None:
            for i in range(len(boxes)):
                cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
                cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
        return img
            
    

def write_detection(plt, filename):
    plt.savefig(filename)


def object_detection_by_frame(filename, threshold=0.7, rect_th=2, text_size=2, text_th=2):
    img = cv2.imread(filename)
    boxes, pred_cls = get_prediction(filename, threshold)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if boxes != -1 or pred_cls != -1:
        for i in range(len(boxes)):
            cv2.rectangle(img, boxes[i][0], boxes[i][1], color = (0, 255, 0), thickness = rect_th)
            cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness = text_th)

    plt.figure(figsize=(16,9))
    plt.imshow(img)
    return plt


def detect_all_frames(input, output):
    i = 0
    for filename in sorted(glob.glob(input + '/*.jpg'), key = os.path.getmtime):
        print(filename)
        plt = object_detection_by_frame(filename)
        write_detection(plt, output + '/frame' + str(i) + '.jpg')
        i += 1
        plt.close()


# detectes and writes to 'detected_output' directory
def detect_and_write(filename):
    print('detecting: ' , filename)
    plt = object_detection_by_frame(filename)
    write_detection(plt, filename)
    plt.close()
    print('detection complete: ' , filename)

def threaded_detection(inputDir, outputDir):
    i = 0
    pool = Pool(processes = 1)
    files = sorted(glob.glob(inputDir + '/*.jpg'), key = os.path.getmtime)
    pool.map(detect_and_write, files)
