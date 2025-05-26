from flask import Flask, render_template, send_file
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import random  # Optional, for cache busting

# Tell Flask to use the 'styles' folder for templates
app = Flask(__name__, template_folder='styles')
model = load_model('generator.h5')

def generate_image():
    noise = tf.random.normal([1, 128])
    generated = model(noise, training=False)[0]
    image = (generated * 127.5 + 127.5).numpy().astype(np.uint8)
    image = Image.fromarray(image.squeeze(), mode='L')
    
    os.makedirs('static', exist_ok=True)
    image_path = 'static/generated.png'
    image.save(image_path)
    return image_path

@app.route('/')
def index():
    image_path = generate_image()
    return render_template('index.html', image_path=image_path, random=random.random)

@app.route('/image')
def image():
    return send_file('static/generated.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
