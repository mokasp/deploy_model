#!/usr/bin/env python3
import cv2
from prediction_input import load_data, normalize_vectors
from lab_to_rgb import lab_to_rgb
from make_grid import make_grid
from output_image import output_image

def display_colors(image):

    X, y = load_data(image)

    X_rgb = lab_to_rgb(X)
    y_rgb = lab_to_rgb(y)

    img_array = make_grid(X_rgb)
    output = output_image(y_rgb)
    print(img_array)
    print(img_array.shape)

    return img_array, output

