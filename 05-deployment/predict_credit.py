import pickle
from flask import Flask
from flask import request
from flask import jsonify
from waitress import serve

app = Flask('credit_card')

model_file_path = "model1.bin"
dv_file_path = "dv.bin"

with open(model_file_path, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file_path, 'rb') as f_in:
    dv = pickle.load(f_in)

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    result = {
        "churn_probability": round(float(y_pred), 3)
    }
    return jsonify(result)

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=9696)

