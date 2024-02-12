import joblib
import json
import logging
import pandas as pd
import pickle

from flask import Flask, request, render_template, json
from typing import Tuple


app = Flask(__name__)

@app.route('/', methods=['Get', 'POST'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['Get', 'POST'])
def predict():
    try:
        upload_file = request.files['file']

        df = pd.read_csv(upload_file)

        with open("trained_model.pkl", "rb") as file:
            model = joblib.load(file)

        result = model.predict(df)

        df["prediction"] = result
        return df.to_json(orient="split")
    except Exception as e:
        return json.dumps(f"Error: {e}")
    
if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)