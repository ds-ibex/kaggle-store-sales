# Standard imports
import gc
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle


def load_model(path:Path, name:str):
    """
    Load a previously saved pickled model
    Args:
        path (string): path where the model is saved
        name (string): name of the model you want to loan
    Returns:
        model: loaded model
    """
    model_path=os.path.join(path.resolve().as_posix(), name)
    model = pickle.load(open(model_path, 'rb'))
    return model

def save_model(path, model, name):

    """
    Save a model under the specified location in a pickle format
    Args:
        path (string): path where the model is saved
        model: model to save
        name (string): name of the model you want to loan
    Returns:
        
    """
    model_path=os.path.join(path.resolve().as_posix(), name)
    pickle.dump(model, open(model_path, 'wb'))
    print("Model has been saved in the following location",model_path)
    return()