#!/usr/bin/env python3
import os
import numpy as np
import cv2
import mediapipe as mp


def direction(image, face_mesh, segmentation):

    # Load the image
    img = image
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run segmentation
    segmentation_results = segmentation.process(rgb_img)
    binary_mask = (segmentation_results.segmentation_mask > 0.1).astype(np.uint8)

    # Run face mesh
    face_results = face_mesh.process(rgb_img)

    # Image dimensions
    h, w, _ = img.shape

    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0]

        # Extract the necessary landmarks (nose tip, left eye, right eye)
        nose_tip = landmarks.landmark[2]
        left_eye = landmarks.landmark[133]
        right_eye = landmarks.landmark[33]

        # Convert to pixel coordinates
        fx = int(nose_tip.x * w)
        fy = int(nose_tip.y * h)
        le_x = int(left_eye.x * w)
        le_y = int(left_eye.y * h)
        re_x = int(right_eye.x * w)
        re_y = int(right_eye.y * h)

        # Calculate the horizontal distance between the nose tip and each eye
        nose_to_left_eye = abs(fx - le_x)
        nose_to_right_eye = abs(fx - re_x)

        # Calculate vertical distances (y-coordinates)
        nose_to_left_eye_y = abs(fy - le_y)
        nose_to_right_eye_y = abs(fy - re_y)

        # Set thresholds for detecting left, right, and forward
        horizontal_threshold = 30
        vertical_threshold = 40    # Threshold for y-coordinates to distinguish tilt

        # Determine face orientation based on distances
        if nose_to_left_eye > horizontal_threshold and nose_to_left_eye > nose_to_right_eye and nose_to_left_eye_y < vertical_threshold:
            orientation = "LEFT"
        elif nose_to_right_eye > horizontal_threshold and nose_to_right_eye > nose_to_left_eye and nose_to_right_eye_y < vertical_threshold:
            orientation = "RIGHT"
        else:
            # Set thresholds for detecting left, right, and forward
            horizontal_threshold = 30  # Can be adjusted based on your images
            vertical_threshold = 30  # Threshold for y-coordinates to distinguish tilt
            # Checking if the face is mostly forward (both horizontal and vertical distances are small)
            if nose_to_left_eye < horizontal_threshold and nose_to_right_eye < horizontal_threshold and \
              nose_to_left_eye_y < vertical_threshold and nose_to_right_eye_y < vertical_threshold:
                orientation = "FORWARD"
            else:
                # If face is slightly turned but not clear, assume forward-facing
                orientation = "LEFT"

        return orientation



def combine_lab_values(top_lab, bottom_lab):
    lab_combined = [
        (top_lab[0] + bottom_lab[0]) / 2,
        (top_lab[1] + bottom_lab[1]) / 2,
        (top_lab[2] + bottom_lab[2]) / 2]

    true_L, true_a, true_b = lab_combined

    # ensuring that the values are within proper range
    true_L = np.clip(true_L, 0, 100)
    true_a = np.clip(true_a, -128, 127)
    true_b = np.clip(true_b, -128, 127)

    return [true_L, true_a, true_b]


def get_lab_vector(image):
    # Load your skin patch (e.g. a cheek or forehead crop)
    img = image

    # Convert the image to Lab color space
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Extract L, a, b channels
    L_channel = lab_img[:, :, 0]
    a_channel = lab_img[:, :, 1]
    b_channel = lab_img[:, :, 2]

    # Calculate the mean for each channel
    mean_L = np.mean(L_channel)
    mean_a = np.mean(a_channel)
    mean_b = np.mean(b_channel)

    # Convert to human-interpretable Lab scale
    true_L = mean_L * (100 / 255)  # Scale L to 0-100
    true_a = mean_a - 128  # Center a to [-128, 127]
    true_b = mean_b - 128  # Center b to [-128, 127]

    # Output the Lab color
    true_L = np.clip(true_L, 0, 100)
    true_a = np.clip(true_a, -128, 127)
    true_b = np.clip(true_b, -128, 127)
    lab_vector = [true_L, true_a, true_b]

    return lab_vector



def extract_hair(image, dir, face_mesh, segmentation):
    input_regions = []
    region_names = ["00_side_hair", "01_top_hair"]


    # Load the image
    img = image
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run segmentation
    segmentation_results = segmentation.process(rgb_img)
    binary_mask = (segmentation_results.segmentation_mask > 0.1).astype(np.uint8)

    # Run face mesh
    face_results = face_mesh.process(rgb_img)

    # Image dimensions
    h, w, _ = img.shape

    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0]
        forehead = landmarks.landmark[10]

        fx = int(forehead.x * w)
        fy = int(forehead.y * h)

        ### --- Top Hair Box ---
        top = max(fy - 35, 0)
        bottom = max(fy - 20, 0)
        left = int(w * 0.3)
        right = int(w * 0.6)



        if dir == 'RIGHT':
            top_s = max(fy - 15, 0)
            bottom_s = max(fy - 30, 0)
            left_s = int(w * 0.2)
            right_s = int(w * 0.4)
        else:
            top_s = max(fy - 15, 0)
            bottom_s = max(fy - 30, 0)
            left_s = int(w * 0.5)
            right_s = int(w * 0.7)


        top_hair = img[top:bottom, left:right]
        top_mask = binary_mask[top:bottom, left:right]

        side_hair = img[top:bottom, left_s:right_s]
        side_mask = binary_mask[top:bottom, left_s:right_s]

        input_regions.append(top_hair)
        input_regions.append(side_hair)


    return input_regions, region_names



def extract_regions(image, input_regions, region_names, face_mesh):
    output_regions = []

    # Load Image
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process Image
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape

            # Define Region Indices
            regions = {
                "02_left_eyelid": [225, 30, 29, 27],
                "03_right_eyelid": [445, 260, 259, 257],
                "04_nose_left_side": [189, 221, 193, 55],
                "05_nose_right_side": [413, 441, 417, 285],
                "06_nose_bridge": [197, 196, 195, 419],
                "07_nose_tip": [5, 281, 275, 1, 45, 51, 5],
                "08_forehead": [109, 10, 338, 299, 9, 69, 109],
                "09_left_cheek": [119, 100, 142, 203, 206, 207, 187, 117, 118, 119],
                "10_right_cheek": [348, 329, 371, 423, 426, 427, 411, 346, 347, 348],
                "11_chin": [175, 396, 428, 421, 313, 18, 83, 201, 208, 171, 175],
                "12_top_lips": [81, 82, 13, 312, 311, 267, 0, 37, 81],
                "13_bottom_lips": [181, 84, 17, 314, 405, 403, 317, 14, 87, 181]
            }


            # Create separate cropped images of each region
            for name, indices in regions.items():
                # Get the coordinates for the region
                pts = np.array([(int(face_landmarks.landmark[i].x * w),
                                int(face_landmarks.landmark[i].y * h)) for i in indices])

                # Compute bounding box
                x, y, w_box, h_box = cv2.boundingRect(pts)
                padding = 0
                x = max(0, x - padding)
                y = max(0, y - padding)
                w_box = min(image.shape[1] - x, w_box + 2 * padding)
                h_box = min(image.shape[0] - y, h_box + 2 * padding)

                # Crop the region
                cropped_region = image[y:y+h_box, x:x+w_box]
                cropped_region = cv2.resize(cropped_region, (0, 0), fx=10, fy=10)
                region_names.append(name)
                # Save the cropped region
                if name == "12_top_lips" or name == "13_bottom_lips":
                    output_regions.append(cropped_region)
                else:
                    input_regions.append(cropped_region)

    return input_regions, output_regions, region_names



def process_regions(image, hair_face_mesh, segmentation, face_mesh):
    lab_vectors = []
    lip_lab_vectors = []


    dir = direction(image, face_mesh, segmentation)
    input_regions, region_names = extract_hair(image, dir, hair_face_mesh, segmentation)

    input_regions, output_regions, region_names = extract_regions(image, input_regions, region_names, face_mesh)

    j = 0
    for region in input_regions:
        lab_vector = get_lab_vector(region)
        lab_vectors.append(lab_vector)
        j += 1
    for region in output_regions:
        lab_vector = get_lab_vector(region)
        lip_lab_vectors.append(lab_vector)
        j += 1
    if len(lab_vectors) != 12:
        return [], []
    combined_lab_value = combine_lab_values(lip_lab_vectors[0], lip_lab_vectors[1])

    return lab_vectors, combined_lab_value


def load_data(image):
    X = []
    y = []

    # initialize the models for hair segments
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    mp_face_mesh = mp.solutions.face_mesh
    segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    hair_face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # initialize MediaPipe Face Mesh for facial regions
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    input_vector, target_vector = process_regions(image, hair_face_mesh, segmentation, face_mesh)

    input_array = np.array(input_vector)
    if input_array.shape == (12, 3):
        X.append(input_array)
        y.append(target_vector)

    for idx, x in enumerate(X):
        flat = np.array(x).flatten()
        if flat.shape != (36,):
            print(f"Issue at index {idx}: shape before flattening = {np.array(x).shape}, after flattening = {flat.shape}")

    X = np.array(X, dtype=np.float32)  # shape (num_samples, 36)
    y = np.array(y, dtype=np.float32)  # shape (num_samples, 3)

    # dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(buffer_size=len(X)).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
    return X, y



def normalize_vectors(data, loaded_data=False):
    if loaded_data:
        X = data['X']
        y = data['y']
    else:
        X = data[0]
        y = data[1] 

    # normalize input Lab vectors)
    X_normalized = []
    for x in X:
        # iterate over each region
        normalized_X = []
        for region in x:
            true_L, true_a, true_b = region
            normalized_L = true_L / 100.0  # Normalize L to [0, 1]
            normalized_a = (true_a + 128) / 255.0  # Normalize a to [0, 1]
            normalized_b = (true_b + 128) / 255.0  # Normalize b to [0, 1]
            normalized_X.append([normalized_L, normalized_a, normalized_b])

        X_normalized.append(np.array(normalized_X))  # adding normalized regions to X_normalized

    X_normalized = np.array(X_normalized)

    # normalize target Lab vectors
    y_normalized = []
    for target in y:
        true_L, true_a, true_b = target
        normalized_L = true_L / 100.0  # normalize L to [0, 1]
        normalized_a = (true_a + 128) / 255.0  # normalize a to [0, 1]
        normalized_b = (true_b + 128) / 255.0  # normalize b to [0, 1]
        y_normalized.append([normalized_L, normalized_a, normalized_b])

    y_normalized = np.array(y_normalized)

    return X_normalized, y_normalized
