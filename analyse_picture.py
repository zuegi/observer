import cv2
import sys, os
from imutils import build_montages
import face_recognition
import numpy as np
import pickle
import h2o
from h2o.estimators import H2OKMeansEstimator

winname = "Test"
cv2.namedWindow(winname)  # Create a named window
cv2.moveWindow(winname, 2000, 200)  # Move it to (40,30)

cv2.setWindowProperty(winname, cv2.WND_PROP_TOPMOST, 1)


def show(img):
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


cwd = os.path.dirname(sys.argv[0])
images =pickle.loads(open(cwd +"/faces.pickle", "rb").read())
# print(img1.shape)
# print(img2.shape)
#
# # scale_percent = 30 # percent of original size
# # width = int(img1.shape[1] * scale_percent / 100)
# # height = int(img1.shape[0] * scale_percent / 100)
# # dim = (width, height)
#
dim = (96, 96)
#
# # resize image
# resized_img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
# print(resized_img1.shape)
# resized_img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
# print(resized_img2.shape)
for image in images:
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    image = image/255.0
    image.reshape(len(image),-1)

print("images type: ", type(images))

# h2o stuff with kmeans
h2o.init(ip="localhost", port=54321)

# wandle die images in ein h2o frame
img_frame = h2o.H2OFrame(images)

# split in train und valid frames
train_frame, valid_frame = img_frame.split_frame(ratios=[.8], seed=1234)


# build the model
fc_model = H2OKMeansEstimator(
    # estimate_k=True,      # Kmeans soll die Anzahl Cluster eruieren
    k=7,                        # maximal 8 Cluster sollen definiert werden
    standardize=False,          # numerischen Spalten haben einen Mittelwert von Null und eine Einheitsvarianz
    max_iterations=1000,          # Anzahl der Trainings-Iterationen, default = 10
    # score_each_iteration=True,  # Jede Iteration wird bewertet - default false
    seed=1234,                  # random number generator seed
    # nfolds = 5                  # nfold +1 model werden verwendet um das modell zu prüfen
)

# fc_model.train(x=list(range(10)), training_frame=train_frame)
fc_model.train(x=["C1"], training_frame=train_frame)
#
# damit kriegen wir die Cluster Zentren
centroids = fc_model.centers()
print("centroids type: ", type(centroids))
print("centroids: ", len(centroids))

# montages = build_montages(images, dim, (5, 5))
# for montage in montages:
#     cv2.namedWindow(winname)  # Create a named window
#     cv2.moveWindow(winname, 2000, 200)  # Move it to (40,30)
#     cv2.imshow(winname, montage)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
