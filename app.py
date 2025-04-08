from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import os

# Load model
model = tf.keras.models.load_model('lstm_autoencoder.h5')

# Optional: Load scaler if used during training
scaler = None
if os.path.exists('scaler.pkl'):
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return 'LSTM Anomaly Detection API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = np.array(data['input'])

        # Optional: Apply scaler
        if scaler is not None:
            original_shape = input_data.shape
            input_data = input_data.reshape(-1, input_data.shape[-1])
            input_data = scaler.transform(input_data)
            input_data = input_data.reshape(original_shape)

        # Predict reconstruction
        reconstructed = model.predict(input_data)
        reconstruction_error = np.mean(np.abs(input_data - reconstructed), axis=(1, 2))

        # You can set your own threshold
        threshold = 0.05  # example value, adjust as needed
        predictions = (reconstruction_error > threshold).astype(int)

        return jsonify({
            'reconstruction_error': reconstruction_error.tolist(),
            'abnormal': predictions.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
