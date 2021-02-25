import cv2
import os, sys
import face_recognition

cwd = os.path.dirname(sys.argv[0])
# main
images = face_recognition.load_images_from_folder(cwd +"/training/images/face-detetction")
print("Found {0} images!".format(len(images)))

for image in images:
    print(image.shape)
