from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model and scaler
model = load_model('./model/content/model2')
scaler = joblib.load('./scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = np.array([request.json['data']])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        return jsonify(prediction.tolist())
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/', methods=['GET'])
def home():
    return 'HELLO'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)