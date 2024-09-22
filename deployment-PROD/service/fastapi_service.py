from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from utils.deployment.model_loader import LoadModel, LoadPipeline

from utils.data_collection.classes import PropertyModel

model = LoadModel()
pipeline = LoadPipeline()
pipeline.named_steps['density'].stage = 'deploy'

# APP
app = FastAPI()
# Permitir solicitudes desde cualquier origen
app.add_middleware(CORSMiddleware, allow_origins=['*'])

@app.get('/')
def main_page():
    return "REST service is activa via FastApi"

@app.post('/model/predict')
def predict(request: PropertyModel):
    data = {'return': False}
    
    request = request.model_dump()
    
    if isinstance(request, dict):
        sample = pd.DataFrame(request, index=[0])
        if 'id' not in sample.columns:
            sample['id']= 0
        transformed_sample = pipeline.transform(sample).drop('price',axis=1)
        pred = model.predict(transformed_sample)[0].round(-2)

        data['success'] = True
        data['response'] = pred
    
    return data['response']