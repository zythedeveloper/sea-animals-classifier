from flask import Flask, request, redirect, render_template, url_for
from classifier import Classifier
import tensorflow as tf
import json
import csv

efficientnet_dir = "weights/efficientnet-weights.keras"
densenet_dir = "weights/densenet-weights.keras"
portNum = 3000

TARGET_SIZE = (224, 224)
INPUT_SIZE = (224, 224, 3)
LABELS = {0: 'Clams', 1: 'Corals', 2: 'Crabs', 3: 'Dolphin', 4: 'Eel', 5: 'Fish', 6: 'Jelly Fish', 7: 'Lobster', 8: 'Nudibranchs', 9: 'Octopus', 10: 'Otter', 11: 'Penguin', 12: 'Puffers', 13: 'Sea Rays', 14: 'Sea Urchins', 15: 'Seahorse', 16: 'Seal', 17: 'Sharks', 18: 'Shrimp', 19: 'Squid', 20: 'Starfish', 21: 'Turtle_Tortoise', 22: 'Whale'}

# initialize the models
global model_eff
global model_dense

print("Initializing the model...")
model_eff = Classifier(INPUT_SIZE, TARGET_SIZE, LABELS,
                        architecture=tf.keras.applications.efficientnet.EfficientNetB3,
                        preprocess_func=tf.keras.applications.efficientnet.preprocess_input,)
model_eff.create_and_load_model(weight_dir=efficientnet_dir)

model_dense = Classifier(INPUT_SIZE, TARGET_SIZE, LABELS,
                        architecture=tf.keras.applications.DenseNet201,
                        preprocess_func=tf.keras.applications.densenet.preprocess_input,)
model_dense.create_and_load_model(weight_dir=densenet_dir)

app = Flask(__name__)

@app.route('/', methods=["GET"])
def home():
    return render_template("index.html")

@app.route('/predict', methods=["GET"])
def predict():
    return render_template("predict.html")

@app.route('/predict', methods=["POST"])
def predict_image():
    global pred_eff_str
    global pred_dense_str
    global image_path

    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    try:
        res_eff = model_eff.predict(image_path).get_json() 
        res_dense =  model_dense.predict(image_path).get_json() 
    except:
        return render_template("predict.html", error="request failed.")
    
    pred_eff_str = [obj["label"] for obj in res_eff["predictions"]]
    pred_dense_str = [obj["label"] for obj in res_dense["predictions"]]

    return render_template(
        "predict.html", 
        labels=LABELS, 
        pred_eff=res_eff["predictions"],
        pred_dense=res_dense["predictions"],
    )

@app.route('/submit', methods=["POST"])
def submit_result():
    # Process the form data here
    form_data = request.form
    actual_label = LABELS[int(form_data.get('actual_label'))]

    # Open the CSV file in append mode
    with open('results/efficientnet.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([image_path, actual_label, actual_label==pred_eff_str[0], actual_label in pred_eff_str])
    file.close()

    with open('results/densenet.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([image_path, actual_label, actual_label==pred_dense_str[0], actual_label in pred_dense_str])
    file.close()

    return render_template("predict.html")

if __name__ == '__main__':
    app.run(port=portNum, debug=True)