import json
import os
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd




# --sauvegardes--

def save_model_results(file_path, model_name, params, nrmse, mape, mae, r2):
    try:
        with open(file_path, "r") as file:
            results = json.load(file)
    except FileNotFoundError:
        results = []
    
    results.append({
        "Model": model_name,
        "Parameters": params,
        "NRMSE": nrmse,
        "MAPE": mape,
        "MAE": mae,
        "R²": r2
    })
    
    with open(file_path, "w") as file:
        json.dump(results, file, indent=4)


def calculate_metrics(y_true, y_pred):
    nrmse = mean_squared_error(y_true, y_pred, squared=False) / (y_true.max() - y_true.min())
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)  #
    r2 = r2_score(y_true, y_pred)
    return nrmse, mape, mae, r2

# --online approch--

def data_stream(df, chunk_size):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]

