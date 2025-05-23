#!/usr/bin/env python3
from prediction_input import load_data, normalize_vectors
from lab_to_rgb import lab_to_rgb
from make_grid import make_grid
from output_image import output_image
import logging
import requests

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

def call_model_api(processed_input):
    url = 'https://model-only.onrender.com/predict'
    payload = {'input': processed_input.tolist()}
    response = requests.post(url, json=payload)
    return response.json().get('prediction')
