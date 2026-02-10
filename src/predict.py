from .train import transform_binary, transform_percent, transform_freq

import joblib

from pathlib import Path
import numpy as np



'''
def get_latest_model_path():
    """Find and return the most recent model file"""
    artifacts_dir = #Path(__file__).parent.parent / "artifacts"
    model_files = list(artifacts_dir.glob("model_v*.joblib"))
    if not model_files:
        raise FileNotFoundError("[ERROR] Aucun modèle versionnélé trouvé dans artifacts/")
    # Sort by version number and return the latest
    return sorted(model_files, key=lambda f: int(f.stem.split('_v')[1]))[-1]
'''

def load_model():
    model_path = "./model/ml_model.joblib"
    # Add functions to __main__ module so pickle can find them during unpickling
    import sys
    main_module = sys.modules['__main__']
    setattr(main_module, 'transform_binary', transform_binary)
    setattr(main_module, 'transform_percent', transform_percent)
    setattr(main_module, 'transform_freq', transform_freq)
    print("#############################################")
    print("model loaded")
    return joblib.load(model_path)



# def predict(model, age: float, salary: float):
#     X = np.array([[age, salary]])
#     return model.predict(X)[0]

def predict(model,X):    
    return model.predict(X)[0]