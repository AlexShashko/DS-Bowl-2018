import os
import sys

import numpy as np

from skimage.io import imread
from skimage.transform import resize

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3


def get_train_data():
    train_ids = next(os.walk("stage1_train"))[1]

    x_train_f = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    y_train_f = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for n, img_id in enumerate(train_ids):
        path = "./stage1_train/" + img_id
        img = imread(path + "/images/" + img_id + ".png")[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True)
        x_train_f[n] = img

        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_id in os.listdir(path + "/masks/"):
            mask_file = imread(path + "/masks/" + mask_id)
            mask_file = np.expand_dims(resize(mask_file, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True),
                                       axis=-1)
            mask = np.maximum(mask, mask_file)
        y_train_f[n] = mask

        if n % 100 == 0 and n != 0 or n == 669:
            print(f"Load {n} images")

    return x_train_f, y_train_f


def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)


def get_UNet_model(filters_dim: list, activation="relu", kernel_init="he_normal", padding="same"):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    new_inputs = Lambda(lambda x: x / 255)(inputs)
    conv_layers = []

    # encoder
    for i in range(len(filters_dim) - 1):
        conv = Conv2D(filters_dim[i], (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_init)(
            new_inputs)
        conv = Conv2D(filters_dim[i], (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_init)(
            conv)
        conv_layers.append(conv)
        new_inputs = MaxPooling2D(pool_size=(2, 2))(conv)

    # bridge
    conv = Conv2D(filters_dim[-1], (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_init)(
        new_inputs)
    conv = Conv2D(filters_dim[-1], (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_init)(conv)
    new_inputs = Dropout(0.5)(conv)

    filters_dim.reverse()
    conv_layers.reverse()

    # decoder
    for i in range(1, len(filters_dim)):
        up = Conv2DTranspose(filters_dim[i], (2, 2), strides=(2, 2), padding=padding)(new_inputs)
        concat = concatenate([up, conv_layers[i - 1]])
        conv = Conv2D(filters_dim[i], (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_init)(
            concat)
        new_inputs = Conv2D(filters_dim[i], (3, 3), activation=activation, padding=padding,
                            kernel_initializer=kernel_init)(conv)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(new_inputs)
    model = Model(inputs=[inputs], outputs=[outputs], name="UNet")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_coefficient])
    return model


def fit_model(model_file="model-weights.h5"):
    unet = get_UNet_model([16, 32, 64, 128, 256])
    x_train, y_train = get_train_data()

    early_stopper = EarlyStopping(patience=5, verbose=1)
    check_pointer = ModelCheckpoint(model_file, verbose=1, save_best_only=True)

    unet.fit(x_train, y_train, validation_split=0.1, batch_size=16, epochs=50,
                       callbacks=[early_stopper, check_pointer])

if __name__ == "__main__":
    fit_model()