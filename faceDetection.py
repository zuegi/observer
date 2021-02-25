import cv2 as cv
import os
import face_recognition as face_recognition
from imutils import build_montages


# main
images = face_recognition.load_images_from_folder(os.getcwd() +"/images/face-detetction")
print("Found {0} images!".format(len(images)))

# initialize the list of images
montage_images = face_recognition.recognize_faces(images)

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
