import os
import sys
import numpy as np
import cv2 as cv
from imutils import build_montages

faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eyesCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
smileCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_smile.xml")
# frontCatCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalcatface_extended.xml")
# frontFaceAltCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt.xml")
# cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt2.xml")
# cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt_tree.xml")
# cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
# cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_fullbody.xml")
# cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_lefteye_2splits.xml")
# cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_licence_plate_rus_16stages.xml")
# cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_lowerbody.xml")
# cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_profileface.xml")
# cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_righteye_2splits.xml")
# cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_russian_plate_number.xml")
# cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_upperbody.xml")

# load images from directory
def load_images_from_folder(folder):
    images = []
    for subfolder in os.listdir(folder):
        dir = folder +"/" +subfolder
        if(os.path.isdir(dir)):
            for filename in os.listdir(dir):
                img = cv.imread(os.path.join(dir, filename))
                if img is not None:
                    images.append(img)
    return images


# main
images = load_images_from_folder(os.getcwd() +"/images/face-detetction")
print("Found {0} images!".format(len(images)))

# initialize the list of images
montage_images = []

for image in images:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # https://stackoverflow.com/questions/36218385/parameters-of-detectmultiscale-in-opencv-using-python
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    # print("Found {0} faces!".format(len(faces)))
    for (x,y,w,h) in faces:
        # img = cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        # roi_color = img[y:y+h, x:x+w]

        eyes = eyesCascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(10, 10))
        # for (ex,ey,ew,eh) in eyes:
            # img_eyes = cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)


        smiles = smileCascade.detectMultiScale(roi_gray, minNeighbors=20)
            # for(sx, sy, sw, sh) in smiles:
            #     img_smile = cv.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (0,0,255), 3)

        # when faces and eyes are found append the image to array
        if(len(faces) >0 and len(eyes) >0 and len(smiles) >0):
            montage_images.append(image)

print("Found {0} image with faces!".format(len(montage_images)))
# construct the montages for the images
# https://www.pyimagesearch.com/2017/05/29/montages-with-opencv/
montages = build_montages(montage_images, (96, 96), (5, 5))

# loop over the montages and display each of them
for montage in montages:
    cv.imshow("Montage", montage)
    cv.waitKey(0)
    cv.destroyAllWindows()
# kann ich die gefunden Images nun Clustern mit kmeans?
# so etwas http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/estimate_k.html


