#### by Nick Speer

## Setup
#### Make sure python3 is current and virtualenv is installed
#### Cuda and Cudnn are necessary for GPU acceleration. This current version primarily uses CPU
#### sudo bash setup.sh

## Object Detection
This repo has an object detection module that uses the ResNet model implemented in PyTorch
![object-detection](https://github.com/nspeer12/Jimbo/blob/master/sample-output.jpg)

#### object_detection_by_frame( filename )
Takes an input file such as 'image.jpg' and returns a MatPlotLib plot.

#### write_detection( matplotlib.pyplot, filename )
Takes a matplotlib plot and saves it to the specified filename

#### detect_all_frames( local input directory, local output directory )
Pass in the a local directory containing the images you want to detect and specify the output directory. This function will go through every frame in the input directory and output it as frame<i>.jpg in the output directory. Only pass directories.


## Camera
Tools to operate a camera and prepare video for object detection

#### Manually specify device in camera.py before using

#### record_video( filename , mode )
Records frames from camera device.
mode is set to 'r' for recording by default. Set to 'v' to only view video feed.
Press q to quit.

#### write_to_video( local input directory, video name )
Takes a directory of images and creates a video.
The video file is large.

## Video
#### digest_video( filename , output directory )
Takes a video and separates it into a series of frames.
