# Import Data Science Libraries
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import flask
import io

# Tensorflow Libraries
from tensorflow import keras
from keras import layers,models,preprocessing,Model
from keras.layers import Dense, Dropout
from keras.preprocessing.image import load_img, img_to_array

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

class Classifier:
    def __init__(self, input_size, target_size, labels, architecture, preprocess_func):
        self.model = None
        self.input_size = input_size
        self.target_size = target_size
        self.labels = labels
        self.architecture = architecture
        self.preprocess_func = preprocess_func

    def create_and_load_model(self, weight_dir):
        data_augmentation = tf.keras.Sequential([
            layers.Resizing(*self.target_size),
            layers.Rescaling(1./255),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])

        self.model = self.architecture(
            input_shape=self.input_size,
            include_top=False,
            weights='imagenet',
            pooling='max'
        )

        inputs = self.model.input
        x = data_augmentation(inputs)

        x = Dense(256, activation='relu')(self.model.output)
        x = Dropout(0.45)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.45)(x)

        outputs = Dense(len(self.labels), activation='softmax')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.load_weights(weight_dir)
        print("Model weights loaded")

    def prepare_image(self, path):
        image = load_img(path)

        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess the input image
        image = image.resize(self.target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = self.preprocess_func(image)

        return image
    
    def predict(self, path):
        image = self.prepare_image(path=path)
        
        # Initialize the data dictionary that will be returned from the view
        data = {"success": False}

        pred = self.model.predict(image)[0]
        index = np.argsort(pred)[::-1][:5]

        data["predictions"] = []

        # Loop over the results and add them to the list of returned predictions
        for i in index:
            label = self.labels[i]
            r = {"label": label, "probability": float(pred[i])}
            data["predictions"].append(r)

        # Indicate that the request was a success
        data["success"] = True

        # Return the data dictionary as a JSON response
        return flask.jsonify(data)