import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils import *

HOG = "hog"
SPATIAL = "spatial"
HISTOGRAM = "hist"

SVC_PATH = "./pickle/svc.obj"
X_SCALER_PATH = "./pickle/X_scaler.obj"

def run(params, max_sample_size=None, retrain=True, save=True):

    if not retrain and os.path.isfile(SVC_PATH) and os.path.isfile(X_SCALER_PATH):

        with open(SVC_PATH, "rb") as f:
            svc = pickle.load(f)
        with open(X_SCALER_PATH, "rb") as f:
            X_scaler = pickle.load(f)

        return svc, X_scaler

    # non car image
    noncar_images = glob.glob("non-vehicles/Extras/*.png")
    noncar_images += glob.glob("non-vehicles/GTI/*.png")
    print("{} non-car images...".format(len(noncar_images)))

    # car_images
    # GTI are time-series images
    gti_car_images = glob.glob("vehicles/GTI_Far/*.png")
    gti_car_images += glob.glob("vehicles/GTI_Left/*.png")
    gti_car_images += glob.glob("vehicles/GTI_MiddleClose/*.png")
    gti_car_images += glob.glob("vehicles/GTI_Right/*.png")
    # KITTI image are not time-series
    kitti_car_images = glob.glob("vehicles/KITTI_extracted/*.png")
    car_images = gti_car_images + kitti_car_images
    print("{} car images...".format(len(car_images)))

    # Reduce the sample size
    #max_sample_size = min(max_sample_size, len(car_images))
    car_images = car_images[0:max_sample_size]
    noncar_images = noncar_images[0:max_sample_size]

    t = time.time()

    # init feature arrays
    car_features = np.empty((len(car_images), 0))
    noncar_features = np.empty((len(noncar_images), 0))

    for feature in params['features']:

        if feature == HOG:
            fn = extract_hog_features
        elif feature == SPATIAL:
            fn = extract_bin_spatial
        elif feature == HISTOGRAM:
            fn = extract_color_hist

        # convert to array and stack on the end of existing features
        _features = fn(car_images, **params['features'][feature], to_array=True)
        car_features = np.hstack((car_features, _features))

        _features = fn(noncar_images, **params['features'][feature], to_array=True)
        noncar_features = np.hstack((noncar_features, _features))

    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract: {}'.format(params['features']))
    # Create an array stack of feature vectors

    X = np.vstack((car_features, noncar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    if save:
        with open(SVC_PATH, 'wb') as f:
            pickle.dump(svc, f)
        with open(X_SCALER_PATH, "wb") as f:
            pickle.dump(X_scaler, f)

    return svc, X_scaler

if __name__ == "__main__":
    run()