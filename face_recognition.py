import os
import cv2 as cv
from imutils import paths
# pickle to save the encodings
import pickle
import numpy as np

faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eyesCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
smileCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_smile.xml")


# load images from directory
def load_images_from_folder(folder):
    imagePaths = list(paths.list_images(folder))
    images = []
    for imagePath in imagePaths:
        img = cv.imread(imagePath)
        if img is not None:
            images.append(img)
    return images


def recognize_faces(images):
    montage_images = []
    for image in images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # https://stackoverflow.com/questions/36218385/parameters-of-detectmultiscale-in-opencv-using-python
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # print("Found {0} faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            # img = cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y + h, x:x + w]
            # roi_color = img[y:y+h, x:x+w]

            eyes = eyesCascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(10, 10))
            # for (ex,ey,ew,eh) in eyes:
            # img_eyes = cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)

            smiles = smileCascade.detectMultiScale(roi_gray, minNeighbors=20)
            # for(sx, sy, sw, sh) in smiles:
            #     img_smile = cv.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (0,0,255), 3)

            # when faces and eyes are found append the image to array
            if len(faces) > 0 and len(eyes) > 0 and len(smiles) > 0:
                montage_images.append(image)

    print("Found {0} image with faces!".format(len(montage_images)))
    return montage_images

def recognize_faces_np(images):
    montage_images = []
    for image in images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # https://stackoverflow.com/questions/36218385/parameters-of-detectmultiscale-in-opencv-using-python
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # print("Found {0} faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            # img = cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y + h, x:x + w]
            # roi_color = img[y:y+h, x:x+w]

            eyes = eyesCascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(10, 10))
            # for (ex,ey,ew,eh) in eyes:
            # img_eyes = cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)

            smiles = smileCascade.detectMultiScale(roi_gray, minNeighbors=20)
            # for(sx, sy, sw, sh) in smiles:
            #     img_smile = cv.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (0,0,255), 3)

            # when faces and eyes are found append the image to array
            if len(faces) > 0 and len(eyes) > 0 and len(smiles) > 0:
                montage_images.append(image)

    print("Found {0} image with faces!".format(len(montage_images)))
    np_images = np.array(montage_images)

    return np_images


def save_faces_np(folder):
    images = load_images_from_folder(folder)
    faces = recognize_faces_np(images)
    print("[INFO] serializing {0} images ".format(len(faces)))
    f = open("faces.pickle", "wb")
    f.write(pickle.dumps(faces))
    f.close()


def face_encode_and_save(folder):
    data = []
    image_paths = list(paths.list_images(folder))
    print("[INFO] found {0} images ".format(len(image_paths)))
    # loop over the image paths
    for (i, imagePath) in enumerate(image_paths):
        image = cv.imread(imagePath)
        img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # print("Found {0} faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            roi_img = img[y:y + h, x:x + w]
            eyes = eyesCascade.detectMultiScale(roi_img, scaleFactor=1.05, minNeighbors=5, minSize=(10, 10))
            smiles = smileCascade.detectMultiScale(roi_img, minNeighbors=20)
            if len(faces) > 0 and len(eyes) > 0 and len(smiles) > 0:
                successul, jpeg_frame = cv.imencode(".jpg", image)
                d = [{"imagePath": imagePath, "encoding": jpeg_frame}]

                data.extend(d)

    # dump the facial encodings data to disk
    print("[INFO] serializing {0} encodings ".format(len(data)))
    f = open("encodings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()


def face_encoding(images):
    face_encodings = []
    for image in images:
        successul, jpeg_frame = cv.imencode(".jpg", image)
        if successul is True:
            face_encodings.append(jpeg_frame)
    return face_encodings


def extract_feature_and_save(folder):
    data=[]
    image_paths = list(paths.list_images(folder))
    print("[INFO] found {0} images ".format(len(image_paths)))
    for (i, imagePath) in enumerate(image_paths):
        image = cv.imread(imagePath)
        img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # print("Found {0} faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            roi_img = img[y:y + h, x:x + w]
            eyes = eyesCascade.detectMultiScale(roi_img, scaleFactor=1.05, minNeighbors=5, minSize=(10, 10))
            smiles = smileCascade.detectMultiScale(roi_img, minNeighbors=20)
            if len(faces) > 0 and len(eyes) > 0 and len(smiles) > 0:
                features = extract_features(image)
                d = [{"imagePath": imagePath, "features": features}]
                data.extend(d)

    # dump the facial encodings data to disk
    print("[INFO] serializing {0} images with features ".format(len(data)))
    f = open("image_path_with_features.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()


def extract_histogram_save(folder):
    data=[]
    image_paths = list(paths.list_images(folder))
    print("[INFO] found {0} images ".format(len(image_paths)))
    for (i, imagePath) in enumerate(image_paths):
        image = cv.imread(imagePath)
        img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # print("Found {0} faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            roi_img = img[y:y + h, x:x + w]
            eyes = eyesCascade.detectMultiScale(roi_img, scaleFactor=1.05, minNeighbors=5, minSize=(10, 10))
            smiles = smileCascade.detectMultiScale(roi_img, minNeighbors=20)
            if len(faces) > 0 and len(eyes) > 0 and len(smiles) > 0:
                # extract a 3D RGB color histogram from the image,
                # using 8 bins per channel, normalize, and update
                # the index
                hist = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                                    [0, 256, 0, 256, 0, 256])
                hist = cv.normalize(hist, hist).flatten()
                d = [{"imagePath": imagePath, "histogram": hist}]
                data.extend(d)

    # dump the facial encodings data to disk
    print("[INFO] serializing {0} images with features ".format(len(data)))
    f = open("image_path_with_histogram.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()

def extract_histogram_save(folder):
    data=[]
    histSize = 256
    histRange = (0, 128) # the upper boundary is exclusive
    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/histSize ))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    accumulate = False

    image_paths = list(paths.list_images(folder))
    print("[INFO] found {0} images ".format(len(image_paths)))
    for (i, imagePath) in enumerate(image_paths):
        image = cv.imread(imagePath)
        # img = cv.cvtColor(image)
        faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # print("Found {0} faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            roi_img = image[y:y + h, x:x + w]
            eyes = eyesCascade.detectMultiScale(roi_img, scaleFactor=1.05, minNeighbors=5, minSize=(10, 10))
            smiles = smileCascade.detectMultiScale(roi_img, minNeighbors=20)
            if len(faces) > 0 and len(eyes) > 0 and len(smiles) > 0:

                # extract a 3D RGB color histogram from the image,
                # using 8 bins per channel, normalize, and update
                # the index
                hist = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                # hist = cv.calcHist([roi_img], [0, 1, 2], None, [histSize], histRange)
                hist = cv.normalize(hist, hist).flatten()
                d = [{"imagePath": imagePath, "image": hist}]
                data.extend(d)

    # dump the facial encodings data to disk
    print("[INFO] serializing {0} images with histogram ".format(len(data)))
    f = open("image_path_with_image.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()

def extract_features(image):
    vector_size = 32
    # Using KAZE, cause SIFT, ORB and other was moved to additional module
    # which is adding addtional pain during install
    alg = cv.KAZE_create()
    # Dinding image keypoints
    kps = alg.detect(image)
    # Getting first 32 of them.
    # Number of keypoints is varies depend on image size and color pallet
    # Sorting them based on keypoint response value(bigger is better)
    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    # computing descriptors vector
    kps, dsc = alg.compute(image, kps)
    # Flatten all of them in one big vector - our feature vector
    dsc = dsc.flatten()
    # Making descriptor of same size
    # Descriptor vector size is 64
    needed_size = (vector_size * 64)
    if dsc.size < needed_size:
        # if we have less the 32 descriptors then just adding zeros at the
        # end of our feature vector
        dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    # print("dsc", dsc)
    return dsc
