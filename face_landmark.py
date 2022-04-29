# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 23:24:11 2022
Face landmark detection
ref: https://www.youtube.com/watch?v=SIZNf_Ydplg
press Esc to exit

@author: Win_N
"""
import cv2
import dlib
from imutils import face_utils

cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
print((lStart, lEnd))
print((rStart, rEnd))

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        
        # range(0, 68) is all face landmarks
        # eyes only
        for n in list(range(36,42)) + list(range(42,48)):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)


    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()