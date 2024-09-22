# Developed by Valentin Tafura - e: valentintafura@hotmail.com

# -----------------------
# Loading model from disk
# -----------------------

import joblib
import os

def LoadModel():
    cur_dir=os.getcwd()
    model_dir=os.path.join(os.path.abspath(os.path.join(cur_dir, '..')), 'model\\model.pkl')
    return joblib.load(model_dir)
    
def LoadPipeline():
    cur_dir=os.getcwd()
    pipeline_dir=os.path.join(os.path.abspath(os.path.join(cur_dir, '..')), 'model\\pipeline.pkl')
    
    return joblib.load(pipeline_dir)
    