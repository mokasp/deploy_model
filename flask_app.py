import os
import base64
import cv2
import numpy as np
from flask import Flask, request, render_template
from datetime import datetime as dt
import tensorflow as tf

app = Flask(__name__)


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

                return f'<h2>Image received!</h2><img src="data:image/jpeg;base64,{type(img_base64)}" width="300">'
            else:
                return 'Error decoding image'
        else:
            return 'Invalid image data'
    return render_template('index.html')