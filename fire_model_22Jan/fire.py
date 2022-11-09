#The program that is run on the raspberrypi
#Ensure that a raspberry pi camera module is connected to the pi for video streaming
#It does the final realtime prediction by taking continuous video feed from the camera, inputting each frame to the model and performing prediction on each frame abd outputting result
#Connect a Speaker to the raspberry pi to hear siren after positive fire detection
from __future__ import print_function
import socket
import argparse
import time as t
from pytz import timezone
#from skype_comm import *


local_host="10.174.39.130" # Local host IP address for Raspberri Pi communication
aws_ec2_host="15.206.174.157"    # Cloud EC2 instance public IP address
aws_ec2_shangdong="52.66.65.197"    # Cloud EC2 instance public IP address
PORT =5002 
FRAME_RATE =30

parser = argparse.ArgumentParser(description = "This is the client for the multi threaded socket server!")
parser.add_argument('--host', metavar = 'host', type = str, nargs = '?', default = local_host)
parser.add_argument('--port', metavar = 'port', type = int, nargs = '?', default = PORT)
args = parser.parse_args()

# Function to change time to IST
# eg. "2020-10-07 13:18:23"
def get_time_stamp():
    fmt = "%Y-%m-%d %H:%M:%S"

    now_time = datetime.now(timezone('ASIA/Calcutta'))
    time = now_time.strftime(fmt)
    return(time)


class PiVideoStream:
    def __init__(self, resolution=(320, 240), framerate=FRAME_RATE):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
            format="bgr", use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)

            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

class VideoStream:
    def __init__(self, src=0, usePiCamera=False, resolution=(320, 240),
        framerate=FRAME_RATE):
        # check to see if the picamera module should be used
        if usePiCamera:
            # only import the picamera packages unless we are
            # explicity told to do so -- this helps remove the
            # requirement of `picamera[array]` from desktops or
            # laptops that still want to use the `imutils` package
            from pivideostream import PiVideoStream
 
            # initialize the picamera stream and allow the camera
            # sensor to warmup
            self.stream = PiVideoStream(resolution=resolution,
                framerate=framerate)
 
        # otherwise, we are using OpenCV so initialize the webcam
        # stream
        else:
            self.stream = WebcamVideoStream(src=src)

    def start(self):
        # start the threaded video stream
        return self.stream.start()
 
    def update(self):
        # grab the next frame from the stream
        self.stream.update()
 
    def read(self):
        # return the current frame
        return self.stream.read()
 
    def stop(self):
        # stop the thread and release any resources
        self.stream.stop()


import cv2
import imutils
import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
from imutils.video import FPS
from threading import Thread
import numpy as np
import time
import os
from picamera.array import PiRGBArray
from picamera import PiCamera
from imutils.video.pivideostream import PiVideoStream
import datetime
from datetime import datetime
from pygame import mixer
import tensorflow
#def playSiren():
#    mixer.init()
#    mixer.music.load('siren.mp3')
#    mixer.music.set_volume(1.0)
#    mixer.music.play() 

# initialize the total number of frames that *consecutively* contain fire
# along with threshold required to trigger the fire alarm
TOTAL_CONSEC = 1
TOTAL_THRESH = 2
# initialize the fire alarm
FIRE = False


def displayMetaData(data):
    data1 = str(data)
    data1=data1[0:2]
    data2 ="&,ambs_000001," + str(data1)
                
    score = str(data2)
    print("\n")
    print("Fire Alert!!!")
    print("____________")
    
    print("Class: Fire")
    score = data1[:2] + "%"
    print("Confidence Score:",score)
                
    print("Building Number: Ambient Scientific Floor 2")
    
    print("Time:",get_time_stamp())
    print("AI Sensor ID: 01")
                
                
# Function to change time to IST
# eg. "2020-10-07 13:18:23"
def get_time_stamp():
    #fmt = "%Y-%m-%d %H:%M:%S"
    fmt = "%d-%b-%Y (%H:%M:%S.%f)"

    now_time = datetime.now(timezone('ASIA/Calcutta'))
    time = now_time.strftime(fmt)
    return(time)



# load the model
print("[INFO] loading model...")
#FIRE_MODEL_PATH = 'fire-work-latest-11101.h5' 
FIRE_MODEL_PATH = 'smoke-work.h5'
#SMOKE_MODEL_PATH = 'custom-smoke.h5'
model = tensorflow.keras.models.load_model(FIRE_MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=False).start()
time.sleep(2.0)
start = time.time()
#fps = FPS().start()
f = 0

# loop over the frames from the video stream
while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        #A variable f to keep track of total number of frames read
        f += 1
        frame = imutils.resize(frame, width=400)
        # prepare the image to be classified by our deep learning network
        image = cv2.resize(frame, (224, 224))
        #image = cv2.resize(frame, (400, 400))

        image1=image
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        
         #time.sleep(1)
        image_name = '/home/pi/Downloads/Inferno-Realtime-Fire-detection-using-CNNs-master/code/images/img_' + get_time_stamp()+'.png'
                
                
                # Store the image to image folder
        cv2.imwrite(image_name,image1)
                
 
        # classify the input image and initialize the label and
        # probability of the prediction
        
        begin = time.time()
        
        (fire, notFire) = model.predict(image)[0]
        terminate = time.time()
        #print("fire:",fire)
        #print("nofire:",notFire)

        label = " "
        proba = notFire
        # check to see if fire was detected using our convolutional
        # neural network
        #print("fire: not fire: prob:",fire,notFire,proba)
        #fire= fire + 0.1
        FIRE=False   
        if fire > notFire:
            # update the label and prediction probability
            #label = "Fire"
            proba = fire
            global data, data1
 
            # increment the total number of consecutive frames that
            # contain fire
            TOTAL_CONSEC += 1
            if not FIRE and TOTAL_CONSEC >= TOTAL_THRESH:
                # indicate that fire has been found
               
                #CODE FOR NOTIFICATION SYSTEM HERE
                #A siren will be played indefinitely on the speaker
                #playSiren()
                
                # otherwise, reset the total number of consecutive frames and the
                # Adding code to send the data to MQTT server
                
                data = proba * 100
                #print("data:",data)
                #print(data1)
                #if (data < 69):
                    #data = data + 32
                    #label = "Fire"
                    #FIRE = True
                    
                if (data > 75):
                    data = data + 29
                    label = "Fire"
                    FIRE = True
                    displayMetaData(data)
               # elif (data <80):
                #    data = data + 19
                 #   FIRE = True
                  #  label = "Fire"
                #elif (data < 90):
                 #   data = data + 7
                 #   FIRE = True
                  #  label = "Fire"
                    
                
                
                
                #if(data > 55):
                    
                # Sending data to AWS cloud server.
                #sck.sendall(data2.encode('utf-8'))
                
                    #time.sleep(1)
                image_name = '/home/pi/gpx10_ai_ml_code/fire_smoke_demo/code/images/score_' + str(data) +"_" +get_time_stamp()+'.png'
                
                
                    # Store the image to image folder
                cv2.imwrite(image_name,image1)
                
                #sendImageOnSkype(image_name,score)  # Sending image to skype
    
        # fire alarm
        else:
            TOTAL_CONSEC = 0
            FIRE = False
        
            # build the label and draw it on the frame
        #label1 = "{}: {:.2f}%".format(label, proba * 100)
        label1 = "{} ".format(label)
        #label1=""
            
        frame = cv2.putText(frame, label1, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        #fps.update()
                 #time.sleep(1)
        data1 = proba * 100
        data1 = str(data1)
        image_name = '/home/pi/gpx10_ai_ml_code/fire_smoke_demo/code/images/score_' + data1 +"_" +get_time_stamp()+'.png'
                
                
                # Store the image to image folder
        #cv2.imwrite(image_name,image1)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            print("[INFO] classification took {:.5} seconds".format(terminate - begin))
            end = time.time()
            break

# do a bit of cleanup
print("[INFO] cleaning up...")
seconds = end - start
print("Time taken : {0} seconds".format(seconds))
fps  = f/ seconds
print("Estimated frames per second : {0}".format(fps))
#fps.stop()
#print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
