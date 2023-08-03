from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import base64
import cv2
import io

app = Flask(__name__)

# Function to create dice coefficient
def dice_coef(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

# Function to create dice loss
def dice_loss(y_true, y_pred, smooth=100):
    return -dice_coef(y_true, y_pred, smooth)

# Function to create iou coefficient
def iou_coef(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou

# Load the segmentation model
model = load_model('unet_m.hdf5', compile=False)
model.compile(Adamax(learning_rate= 0.001), loss= dice_loss, metrics= ['accuracy', iou_coef, dice_coef])

# Preprocess the image
def preprocess_image(image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img = cv2.resize(image, (256, 256))
    img = img/255
    img = img[np.newaxis, :, :, : ]

    return img

# Define the home route
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for image segmentation
@app.route('/segment', methods=['POST'])
def segment():
    # Get the uploaded image file from the form
    image_file = request.files['image']

    # Read the image using PIL
    pil_image = Image.open(image_file).convert('RGB')
    open_cv_image = np.array(pil_image)

    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # Perform image preprocessing
    image = preprocess_image(open_cv_image)

    # Perform image segmentation
    segmented_image = model.predict(image)

    # Plot image with mask
    plt.imshow(np.squeeze(np.squeeze(image)))
    plt.imshow(np.squeeze(segmented_image), alpha=0.7)
    plt.axis('off')
    plt.savefig('overlay_chart.png')

    # Save the chart to a buffer
    chart_buffer = io.BytesIO()
    plt.savefig(chart_buffer, format='png')
    chart_buffer.seek(0)

    # Clear the current figure to release memory
    plt.clf()

    # Convert the chart buffer to a data URI
    chart_data_uri = base64.b64encode(chart_buffer.getvalue()).decode('utf-8')

    return render_template('predict.html', chart_data_uri=chart_data_uri)

# Run the Flask application
if __name__ == '__main__':
    app.run()
