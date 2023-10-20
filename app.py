#! /Library/Frameworks/Python.framework/Versions/3.11/bin/python3
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model and scaler
model = load_model('./tf_model')
scaler = joblib.load('./scaler.pkl')
dv = joblib.load('./dv.pkl')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
    input_data = request.form.to_dict()
    input_data['RevLineCr'] = bool(input_data['RevLineCr'])
    input_data['NewBusiness'] = bool(input_data['NewBusiness'])
    input_data['LowDoc'] = bool(input_data['LowDoc'])
    input_data['ApprovalFY'] = int(input_data['ApprovalFY'])
    input_data['Term'] = int(input_data['Term'])
    input_data['CreateJob'] = int(input_data['CreateJob'])
    input_data['RetainedJob'] = int(input_data['RetainedJob'])
    input_data['NoEmp'] = int(input_data['NoEmp'])
    input_data['ApprovalMonth'] = int(input_data['ApprovalMonth'])
    input_data['DisbursementGross'] = float(input_data['DisbursementGross'])
    input_data = [input_data]
    #return render_template("result.html", prediction = 'none', input_data = input_data)
    try:
        input_data = [request.form.to_dict()]
        transformed_data = dv.transform(input_data)
        scaled_data = scaler.transform(transformed_data)
        result = model.predict(scaled_data)        
        if result >= 0.975 :
            prediction ='Loan will default'
        else:
            prediction ='Loan will be paid in full'           
        return render_template("result.html", prediction = prediction, input_data = input_data)
    except Exception as e:
        return render_template("result.html", prediction = str(e), input_data = input_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
