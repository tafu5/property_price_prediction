# Developed by Valentin Tafura - e: valentintafura@hotmail.com

# -----------------------
# Loading model from disk
# -----------------------

import joblib
import os

cur_dir = os.getcwd()
model_path = os.path.join(cur_dir, 'model')
last_versions = max([int(v) for v in os.listdir(model_path)])

model_version_path = os.path.join(model_path, str(last_versions))

def LoadModel(model_version_path=model_version_path):
    model_dir = os.path.join(model_version_path, 'model.pkl')
    return joblib.load(model_dir)
    
def LoadPipeline(model_version_path=model_version_path):
    
    pipeline_dir = os.path.join(model_version_path, 'pipeline.pkl')
    return joblib.load(pipeline_dir)

def LoadNeighborhoodVar(model_version_path=model_version_path):
    
    neighborhood_var_dir = os.path.join(model_version_path, 'neighborhood_var.pkl')
    return joblib.load(neighborhood_var_dir)
    