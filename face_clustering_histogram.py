import os
import pickle
import sys
import numpy as np
import cv2 as cv
import h2o
from h2o.estimators import H2OKMeansEstimator
from imutils import build_montages

cwd = os.path.dirname(sys.argv[0])
features = pickle.loads(open(cwd +"/image_path_with_histogram.pickle", "rb").read())
print("features type: ", type(features))
feature_list = []
for feature in features:
    feature_list.append(feature['histogram'].tolist())

h2o.init(ip="localhost", port=54321)

# assume you have recognizes the faces and saved it in pickle
# encoded images only

# # wandle die images in ein h2o frame
img_frame = h2o.H2OFrame(feature_list)
numcol = img_frame.ncol
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

# wir wollen nur die Daten in der x Column berechnen lassen und das modell validieren

fc_model.train(x=list(range(10)), training_frame=train_frame)
#
# damit kriegen wir die Cluster Zentren
centroids = fc_model.centers()
print("centroids type: ", type(centroids))
print("centroids: ", len(centroids))
np_centroids = np.asarray(centroids, dtype=np.float32)
for feature in features:
    if np.array_equal(feature, np_centroids):
        cv.imshow("Montage", feature)
        cv.waitKey(0)
        cv.destroyAllWindows()
