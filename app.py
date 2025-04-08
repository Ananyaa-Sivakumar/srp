from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("lstm_model.h5")  # Path to your .h5 file

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_array = np.array(data['input'])  # expects a 2D list (batch, timesteps, features)
    prediction = model.predict(input_array)
    return jsonify({'prediction': prediction.tolist()})
