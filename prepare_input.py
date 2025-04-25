#!/usr/bin/env python3
import cv2
from prediction_input import load_data, normalize_vectors
from lab_to_rgb import lab_to_rgb
from make_grid import make_grid
from output_image import output_image
import logging
from multiprocessing import Process, Queue
import gc


def display_colors(image):

    X, y = load_data(image)

    X_rgb = lab_to_rgb(X)
    y_rgb = lab_to_rgb(y)

    img_array = make_grid(X_rgb)
    output = output_image(y_rgb)

    return img_array, output

def model_input(image, model):
    X, _ = load_data(image)
    logging.debug(X)
    prediction = model.predict(X)

    return prediction

def display_prediction(y_lab):
    y_rgb = lab_to_rgb(y_lab)
    output = output_image(y_rgb)
    return output

def model_runner(img_data, q):
    import tensorflow as tf
    model = tf.keras.models.load_model('model/test_model_00.keras', compile=False)
    prediction = model_input(img_data, model)
    q.put(prediction)

def predict_in_subprocess(img_data):
    q = Queue()
    p = Process(target=model_runner, args=(img_data, q))
    p.start()
    p.join()
    return q.get()
