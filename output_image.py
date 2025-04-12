#!/usr/bin/env python3
import cv2
import numpy as np

def output_image(rgb_vector):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb_vector = rgb_vector[0]
    rgb_vector = rgb_vector[::-1]
    img[:, :] = rgb_vector
    return img