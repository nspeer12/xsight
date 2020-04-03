import os
import pathlib


class ObjectDetection:
    def __init__(self, path_to_model):
        self.model = models.load_model(model_path)
    
    def predict(self, img):
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(img, axis=0))
        boxes /= scale

    def draw_detections(img, boxes, scores, labels):
        for box, score, label in zip(boxes[0], scores[0], labeles[0])
        































