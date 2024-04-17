import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
import keras
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps
import math

text = "Hi Dr/Ghada , I am your assistant , I can help you to classify fruits "
image = None

IMG_SIZE = 244


code_train = {
    "Apple": 0,
    "avocado": 1,
    "Banana": 2,
    "cherry": 3,
    "kiwi": 4,
    "orange": 5,
    "strawberries": 6,
    "Watermelon": 7,
    "pinenapple": 8,
}


def get_image():
    global k, image
    if image is None:
        return "test/test.1.jpeg"
    k = "test/"
    return k + image.__dict__["name"]


def getcode_train(n):
    for x, y in code_train.items():
        if n == y:
            return x


def predict_image():
    # Load the trained model
    model = load_model("keras_Model.h5", compile=False)
    # Print the image file name
    print(get_image())
    # Create an empty numpy array with the shape (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3))
    # Load the image and convert it to RGB mode
    image = Image.open(get_image()).convert("RGB")
    # Resize and crop the image to 224x224
    size = (224, 224) 
    
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    # Convert the image to a numpy array
    image_array = np.asarray(image)
    # Normalize the image array
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # Load the normalized image array into the empty numpy array
    data[0] = normalized_image_array
    # Use the trained model to make a prediction on the image
    prediction = model.predict(data)
    # Find the index of the predicted class with the highest probability
    index = np.argmax(prediction)
    # Get the class name of the predicted class
    class_name = getcode_train(index)
    # Get the confidence score of the predicted class
    confidence_score = prediction[0][index]
    # Round the confidence score to 2 decimal places
    confidence_score = math.floor(confidence_score * 100) / 100
    # Multiply the confidence score by 100 to get a percentage
    confidence_score = (confidence_score) * 100
    # Set the global variable 'text' to a string that includes the predicted class name and confidence score
    global text
    text = "type is " + class_name + " with " + str(confidence_score) + "% confidence"


def get_image():
    global k, image
    if image is None or not hasattr(image, "__dict__") or "name" not in image.__dict__:
        return "test/test.3.jpeg"
    k = "test/"
    return k + image.__dict__["name"]
