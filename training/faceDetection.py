import os
import numpy as np
import cv2 as cv


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
    for filename in os.listdir(folder):
        print("filename: ", filename)
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


# main
print(os.getcwd())
images = load_images_from_folder(os.getcwd() +"/images/face-detetction")

for image in images:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # https://stackoverflow.com/questions/36218385/parameters-of-detectmultiscale-in-opencv-using-python
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x,y,w,h) in faces:
        img = cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eyesCascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(10, 10))
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)

        # smiles = smileCascade.detectMultiScale(roi_gray, minNeighbors=20)
        # for(sx, sy, sw, sh) in smiles:
        #     cv.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (0,0,255), 3)

        cv.imshow('img',image)
        cv.waitKey(0)
        cv.destroyAllWindows()
