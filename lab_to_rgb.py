#!/usr/bin/env python3
import numpy as np
import cv2


def lab_to_rgb(lab_values):
    rgb_vectors = []
    if len(lab_values[0]) == 12:
        for lab_vector in lab_values[0]:
            true_L, true_a, true_b = lab_vector
            lab_color = np.uint8([[[true_L * 255 / 100, true_a + 128, true_b + 128]]])  # Adjust scaling
            rgb_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2BGR)
            rgb_color = rgb_color[0][0]  # Extract the RGB value from the array
            rgb_color_rev = list(rgb_color)
            rgb_color_rev.reverse()
            rgb_color_rev = np.array(rgb_color_rev)
            rgb_vectors.append(rgb_color_rev)
    else:
        true_L, true_a, true_b = lab_values[0]
        lab_color = np.uint8([[[true_L * 255 / 100, true_a + 128, true_b + 128]]])  # Adjust scaling
        rgb_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2BGR)
        rgb_color = rgb_color[0][0]  # Extract the RGB value from the array
        rgb_color_rev = list(rgb_color)
        rgb_color_rev.reverse()
        rgb_color_rev = np.array(rgb_color_rev)
        rgb_vectors.append(rgb_color_rev)
    return rgb_vectors