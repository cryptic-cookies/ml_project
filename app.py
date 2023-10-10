from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model and scaler
model = load_model('./model/content/model')
scaler = joblib.load('./scaler.pkl')

# prediction function
# def ValuePredictor(to_predict_list):
#     to_predict = np.array(to_predict_list).reshape(1, 12)
#     loaded_model = pickle.load(open("model.pkl", "rb"))
#     result = loaded_model.predict(to_predict)
#     return result[0]

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

# @app.route('/result', methods = ['POST'])
# def result():
#     if request.method == 'POST':
#         to_predict_list = request.form.to_dict()
#         to_predict_list = list(to_predict_list.values())
#         to_predict_list = list(map(int, to_predict_list))
#         result = ValuePredictor(to_predict_list)        
#         if int(result)== 1:
#             prediction ='Income more than 50K'
#         else:
#             prediction ='Income less that 50K'           
#         return render_template("result.html", prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
