import numpy as np
import pyscreenshot as Screenshot
from PIL import Image
import cv2
from detection import *
import time




model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()

# define the objects that we can detect
COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'car', 'plane', 'plant']


def get_prediction(img, threshold):
	#img = Image.open(img_path) # Load the image
	transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
	img = transform(img) # Apply the transform to the image
	pred = model([img]) # Pass the image to the model
	pred_class = ['person' for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
	pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
	pred_score = list(pred[0]['scores'].detach().numpy())
	try:
		pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
		pred_boxes = pred_boxes[:pred_t+1]
		pred_class = pred_class[:pred_t+1]
	except:
		print('No predictions made')
		return -1, -1

	return pred_boxes, pred_class



def object_detection(img, threshold=0.8, rect_th=2, text_size=2, text_th=2):
	boxes, pred_cls = get_prediction(img, threshold) # Get predictions

	if (boxes == -1 or pred_cls == -1):
	  return img
	  #img = cv2.imread(img_path) # Read image with cv2
	  #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
	for i in range(len(boxes)):
		cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
		cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class

	return img

def write_detection(plt, filename):
	plt.savefig(filename)


def screen_record():
	i = 0
	while True:
		t = time.time()
		screen =  Screenshot.grab(bbox=(0,0,3840, 2160)) # currently set to 4k resolution
		printscreen_numpy =   np.array(screen.convert('RGB'),dtype='uint8')
		#cv2.imshow('window',cv2.cvtColor(printscreen_numpy, cv2.COLOR_BGR2RGB))
		res = object_detection(cv2.cvtColor(printscreen_numpy, cv2.COLOR_BGR2RGB))
		#cv2.imshow('output', res)
		cv2.imwrite('detections/frame'+str(i)+'.jpg', res)
		print("detected in ", time.time() - t, " seconds")
		i+=1
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

def main():
	# this time.sleep is useful for getting your game ready
	time.sleep(10)
	screen_record()

if __name__=='__main__':
	main()
