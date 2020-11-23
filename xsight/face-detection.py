import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def facial_detection(output_dir='faces/', mode = 'r'):
	faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')	

	i = 0
	while(True):
		ret, image = cap.read()
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
	
		for (x, y, w, h) in faces:
    			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
		
		cv2.imshow('frame', image)

		if( mode == 'r'):
			cv2.imwrite(output_dir + str(i) + '.jpg', image)
			i += 1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break



if __name__=='__main__':
	facial_detection()
