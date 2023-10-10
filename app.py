from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model and scaler
model = load_model('./tf_model')
scaler = joblib.load('./scaler.pkl')
dv = joblib.load('./dv.pkl')


@app.route('/result', methods=['POST'])
def predict():
    try:
        to_predict_list = request.form.to_dict()
        input_data = np.array(to_predict_list) #([request.json['data']])
        #input_data = np.array({'State': {209479: 'IL'},'BankState': {209479: 'IL'},'ApprovalFY': {209479: 2007},'Term': {209479: 40},'NoEmp': {209479: 8},'NewBusiness': {209479: True},'CreateJob': {209479: 2},'RetainedJob': {209479: 8},'UrbanRural': {209479: 'Urban'},'RevLineCr': {209479: True},'LowDoc': {209479: False},'DisbursementGross': {209479: 87570.0},'Default': {209479: 1},'ApprovalMonth': {209479: 5},'NAICSCategory': {209479: 'Other services (except public administration) 92 Public administration'}})
        scaled_data = scaler.transform(dv.transform(input_data))
        prediction = model.predict(scaled_data)
        return jsonify(prediction.tolist())
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
