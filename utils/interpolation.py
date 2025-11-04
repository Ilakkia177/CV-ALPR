# interpolation_utils.py
import pandas as pd
import numpy as np

def interpolate_boxes(csv_path):
    df = pd.read_csv(csv_path)
    all_cols = [c for c in df.columns if "_x" in c or "_y" in c or "_w" in c or "_h" in c]
    for c in all_cols:
        df[c] = df[c].interpolate(method="linear", limit_direction="both")
    return df
