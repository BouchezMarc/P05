from .train import (
    transform_binary,
    transform_freq,
    transform_percent,
)

import joblib


def load_model():
    model_path = "./model/ml_model.joblib"
    # Add functions to __main__ module
    # so pickle can find them during unpickling
    import sys
    main_module = sys.modules["__main__"]
    setattr(main_module, "transform_binary", transform_binary)
    setattr(main_module, "transform_percent", transform_percent)
    setattr(main_module, "transform_freq", transform_freq)
    print("#############################################")
    print("model loaded")
    return joblib.load(model_path)


def predict(model, X):
    return model.predict(X)[0]
