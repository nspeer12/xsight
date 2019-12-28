import cv2

# digest a video and output frames to a directory
def digest_video(filename, outputDir):
	vid = cv2.VideoCapture(filename)
	i = 0
	processing = True

	while(processing):
		processing, img = vid.read()
		cv2.imwrite(outputDir + '/frame%d.jpg' %i, img)
		i +=1
