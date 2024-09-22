import argparse

parser=argparse.ArgumentParser()
parser.add_argument('-p', '--port', required=True, help='Service PORT is required!')
args = parser.parse_args()

# Service port
port = args.port
print("Port recognized: ", port)

from model_loader import LoadModel, LoadPipeline
import pandas as pd

import sys 
import os

model = LoadModel()

# Obtener el directorio padre de DEV y agregarlo al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

pipeline = LoadPipeline()

from flask import Flask, request, jsonify

# Initialize the application service (FLASK)
app = Flask(__name__)

from flask_cors import CORS
CORS(app)

# Define a default route
@app.route('/', methods=['GET'])
def MainPage():
    return "REST service is active via Flask"

# Prediction
@app.route('/model/predict', methods=['POST'])
def prediction():
    data = {'success': False}

    if request.method=='POST':
        if request.is_json:
        
            json_data = request.get_json()

            sample = pd.DataFrame(json_data, index=[0])
            transformed_sample = pipeline.transform(sample).drop('price',axis=1)
            pred = model.predict(transformed_sample)[0].round(-2)

            data['success'] = True
            data['response'] = pred

        else:
            print("Request must be JSON")

    return jsonify(data)

app.run(host='0.0.0.0',port=port, threaded=False) 

