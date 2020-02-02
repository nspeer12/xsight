import numpy as np
import pyscreenshot as Screenshot
from PIL import Image
import cv2
from detection import *





model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()

# define the objects that we can detect
COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person']


def get_prediction(img, threshold):
  #img = Image.open(img_path) # Load the image
  transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  img = transform(img) # Apply the transform to the image
  pred = model([img]) # Pass the image to the model
  pred_class = ['person' for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return pred_boxes, pred_class



def object_detection(img, threshold=0.5, rect_th=3, text_size=3, text_th=3):
  boxes, pred_cls = get_prediction(img, threshold) # Get predictions
  #img = cv2.imread(img_path) # Read image with cv2
  #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
  for i in range(len(boxes)):
    cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
    cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
  plt.figure(figsize=(20,30)) # display the output image
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()

def write_detection(plt, filename):
	plt.savefig(filename)


def screen_record():
	while True:
		# 800x600 windowed mode
		screen =  Screenshot.grab(bbox=(0,40,800,640))
		printscreen_numpy =   np.array(screen.convert('RGB'),dtype='uint8')
		#cv2.imshow('window',cv2.cvtColor(printscreen_numpy, cv2.COLOR_BGR2RGB))
		object_detection(cv2.cvtColor(printscreen_numpy, cv2.COLOR_BGR2RGB))
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

def main():
	screen_record()

if __name__=='__main__':
	main()
