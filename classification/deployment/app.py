import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('EfficientNetB3-Brain Tumors-No.h5')

# Preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    return img_array

# Classes
classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Perform image classification
def classify_image(image):
    image = preprocess_image(image)
    prediction = model.predict(image)
    class_indices = np.argmax(prediction, axis=1)
    class_labels = [classes[i] for i in class_indices]
    probabilities = np.max(prediction, axis=1)

    results = []
    for label, probability in zip(class_labels, probabilities):
        results.append({'label': label, 'probability': float(probability)})

    return results

# Define the threshold for out-of-domain detection
threshold = 0.93

# Flask route for index page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predictAPI', methods=['POST'])
def api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image_file = request.files.get('image')
    image = Image.open(image_file)

    # Convert the image to base64 string
    image_data = io.BytesIO()
    image.save(image_data, format='JPEG')

    results = classify_image(image)

    # Check if the image is out-of-domain
    max_probability = max(result['probability'] for result in results)
    if max_probability < threshold:
        results = [{'label': 'Out-of-domain', 'probability': 1.0}]
    
    return jsonify({'Result': results})


# Flask route for image classification
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image_file = request.files['image']
    image = Image.open(image_file)

    # Convert the image to base64 string
    image_data = io.BytesIO()
    image.save(image_data, format='JPEG')
    encoded_image = base64.b64encode(image_data.getvalue()).decode('utf-8')

    results = classify_image(image)

    # Check if the image is out-of-domain
    max_probability = max(result['probability'] for result in results)
    if max_probability < threshold:
        results = [{'label': 'Out-of-domain', 'probability': 1.0}]

    return render_template('predict.html', image=encoded_image, results=results)

if __name__ == '__main__':
    app.run()
