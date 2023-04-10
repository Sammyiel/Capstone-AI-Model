import io
import os
import random
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
from scipy.special import softmax


app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('crop_disease_model.h5')

# Load the dataset to get the class names
dataset_directory = 'valid'

dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_directory,
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
)
class_names = dataset.class_names


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    img_bytes = io.BytesIO(img.read())
    image = tf.keras.preprocessing.image.load_img(img_bytes, target_size=(256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    predicted_class = np.argmax(predictions[0])

    class_name = class_names[predicted_class]
    similar_images = get_similar_images(class_name)
    
    # Get the confidence percentage
    probabilities = softmax(predictions[0])
    confidence = probabilities[predicted_class] * 100

    return jsonify({"class": class_name, "confidence": confidence, "similar_images": similar_images})


@app.route('/images/<path:filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory(os.path.join(dataset_directory), filename)


def get_similar_images(class_name):
    class_directory = os.path.join(dataset_directory, class_name)
    all_images = os.listdir(class_directory)
    random.shuffle(all_images)
    selected_images = all_images[:5]
    image_paths = [os.path.join(class_name, img) for img in selected_images]
    return image_paths


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
