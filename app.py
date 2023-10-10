from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model and scaler
model = load_model('./model/content/model')
scaler = joblib.load('./model/scaler.pkl')
dv = joblib.load(('./model/dv.pkl')


@app.route('/result', methods=['POST'])
def predict():
    try:
        to_predict_list = request.form.to_dict()
        input_data = np.array(to_predict_list) #([request.json['data']])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        return jsonify(prediction.tolist())
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
