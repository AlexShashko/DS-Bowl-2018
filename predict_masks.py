import os
import sys

import numpy as np

from skimage.io import imread, imsave
from skimage.transform import resize

from keras.models import load_model
from keras import backend as K


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3


def get_test_data():
    test_ids = next(os.walk("stage1_test"))[1]

    x_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    test_img_sizes = []

    for n, img_id in enumerate(test_ids):
        path = "./stage1_test/" + img_id
        img = imread(path +  "/images/" + img_id + ".png")[:, :, :IMG_CHANNELS]
        test_img_sizes.append(img.shape)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True)
        x_test[n] = img

    print("Done.")
    return x_test, test_img_sizes


def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)


def predict(model_file="model-weights.h5"):
    x_test, test_sizes = get_test_data()
    model = load_model(model_file, custom_objects={"dice_coefficient": dice_coefficient})

    predict_test = model.predict(x_test, verbose=0)
    pred_test_mask = (predict_test > 0.5).astype(np.uint8)

    pred_test_unsampled = []
    for i in range(len(pred_test_mask)):
        pred_test_unsampled.append(
            resize(np.squeeze(pred_test_mask[i]), test_sizes[i], mode="constant", preserve_range=True))

    return pred_test_unsampled


if __name__ == "__main__":
    predicted = predict()

    test_ids = next(os.walk("stage1_test"))[1]
    for n, mask in enumerate(predicted):
        imsave("./predicted_test/" + test_ids[i] + ".png", mask)
