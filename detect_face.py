import numpy as np
import cv2
import os, sys

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

cwd = os.path.dirname(sys.argv[0])
img = cv2.imread(cwd +"/training/images/face-detetction/groot/27207FEA-0680-4F28-A2D2-39C6D44D3C0C_1_105_c.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.2, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
       cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    if len(faces) >0 and len(eyes) >0:
        img = roi_color

winname = "Test"
cv2.namedWindow(winname)  # Create a named window
cv2.moveWindow(winname, 2000, 100)  # Move it to (40,30)
cv2.setWindowProperty(winname, cv2.WND_PROP_TOPMOST, 1)
cv2.imshow(winname,img)
cv2.waitKey(0)
cv2.destroyAllWindows()
