# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 22:24:19 2022
Facial Recognition and Liveness Detection through Blink counts
Reference: https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/

This program uses facial landmark detection to identify eyes of a face.
The eye aspect ratio is calculated.
The variable EYE_AR_THRESH = 0.2,
controls the threshold of the EAR that defines a blink.
The variable EYE_AR_CONSEC_FRAMES = 1,
is the number of frames the EAR must be below the threshold to count a blink
During the first 12 seconds of input from the laptop's camera, 
if the frequency of blinks per second is below 0.16 or >= 1.25,
the input is denied and a possible presentation attack is detected.
That is, if there are less than 2 blinks counted in 12 seconds or
more than 14 blinks counted in 12 seconds, 
the input considered a spoof attempt.

@author: Nathan Rice
"""
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from timeit import default_timer as timer
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import copy

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# #ap.add_argument("-p", "--shape-predictor", required=True,
# # 	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
    help="path to input video file")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 1
# initialize the frame counters and the total number of blinks
COUNTER = 0 #total number of successive frames that have an eye aspect ratio less than EYE_AR_THRESH
TOTAL = 0 #total number of blinks that have taken place while the script has been running

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat") #(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
#vs = FileVideoStream(args["video"]).start()
#fileStream = True
vs = VideoStream(src=0).start()
# start timer
start_time = timer()
# vs = VideoStream(usePiCamera=True).start()
#fileStream = False
time.sleep(1.0)


# loop over frames from the video stream
while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    #if fileStream and not vs.more():
    #    break
    # grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=450)#width=450
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
    rects = detector(gray, 0)
	# loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
    	# compute the convex hull for the left and right eye, then
        # vizalize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    	# check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            # otherwise, the eye aspect ratio is not below the blink
            # threshold
        else:
            # if the eyes were closed for a sufficient number of frames
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                    
            #reset the eye frame counter
            COUNTER = 0
                
    # draw the total number of blinks on the frame along with
    # the computed eye aspect ratio for the frame
    cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


    # time duration reached, print accepted or denied
    end_time = timer() # stop timer
    duration = 0
    duration += end_time - start_time
    cv2.putText(frame, "Timer: {:.2f}".format(duration),(10,60),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    blinks_per_sec = TOTAL/duration
    cv2.putText(frame, "BPS: {:.2f}".format(blinks_per_sec),(10,90),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    if duration >= 12 and (blinks_per_sec >= 0.16  and blinks_per_sec < 1.25):
        cv2.putText(frame, "Input Accepted",(130,90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if duration >= 12 and (blinks_per_sec < 0.16 or blinks_per_sec >= 1.25):
        cv2.putText(frame, "Input Denied", (130,100),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, "Possible Presentaion Attack Detected!", (10,130),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        print("Total Blinks: {}".format(TOTAL))
        # stop timer
        elapsed_time = end_time-start_time
        print("Elapsed time: {:.2f}".format(elapsed_time))
        print("Blinks per second: {:.2f}".format(blinks_per_sec))
        # results
        if blinks_per_sec < 0.16 or blinks_per_sec >= 1.25:
            print("Result: FAIL")
        if blinks_per_sec >= 0.16  and blinks_per_sec < 1.25:
            print("Result: PASS")
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
    
    

