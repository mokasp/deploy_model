import os
import base64
import cv2
import numpy as np
from flask import Flask, request, render_template
from datetime import datetime as dt
import tensorflow as tf
import logging
from prepare_input import display_colors, model_input, display_prediction


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
model = tf.keras.models.load_model('model/test_model_00.keras', compile=False)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data_url = request.form['image']  # base64 string

        if data_url.startswith('data:image'):
            header, encoded = data_url.split(",", 1)
            image_data = base64.b64decode(encoded)
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                return f'<h2>Image received!</h2><img src="data:image/jpeg;base64,{img_base64}" width="300">'
            else:
                return 'Error decoding image'
        else:
            return 'Invalid image data'
    return render_template('index.html')

@app.route('/palette', methods=['GET', 'POST'])
def palette():
    if request.method == 'POST':
        data_url = request.form['image']  # base64 string

        if data_url.startswith('data:image'):
            header, encoded = data_url.split(",", 1)
            image_data = base64.b64decode(encoded)
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                img_type = str(type(img_base64))
                logging.debug("🎯 /palette route was hit!")
                logging.debug(f"📸 Image data starts with: {request.form['image'][:30]}")
                logging.debug(f"🧠 Decoded image shape: {img}")
                logging.debug(f"🧠 Decoded image shape: {img.shape}")
                logging.debug(f"🧠 Decoded image type: {img_type}")


                img_array, output = display_colors(img)
                _, buffer = cv2.imencode('.jpg', img_array)
                img_array64 = base64.b64encode(buffer).decode('utf-8')
                _, buffer = cv2.imencode('.jpg', output)
                output64 = base64.b64encode(buffer).decode('utf-8')

                return f'<h2>Image received!</h2><img src="data:image/jpeg;base64,{img_base64}" width="300"><img src="data:image/jpeg;base64,{img_array64}" width="300"><img src="data:image/jpeg;base64,{output64}" width="300">'
            else:
                return 'Error decoding image'
        else:
            return 'Invalid image data'
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data_url = request.form['image']  # base64 string

        if data_url.startswith('data:image'):
            header, encoded = data_url.split(",", 1)
            image_data = base64.b64decode(encoded)
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                img_type = str(type(img_base64))
                logging.debug("🎯 /palette route was hit!")
                logging.debug(f"📸 Image data starts with: {request.form['image'][:30]}")

                prediction = model_input(img, model)
                logging.debug(f"🧠 prediction: {prediction}")
                output = display_prediction(prediction)
                _, buffer = cv2.imencode('.jpg', output)
                output64 = base64.b64encode(buffer).decode('utf-8')


                return render_template('index.html', img_data=data_url, output_img=output64, prediction=prediction)
            else:
                return 'Error decoding image'
        else:
            return 'Invalid image data'
    return render_template('index.html')