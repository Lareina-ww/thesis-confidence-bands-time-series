import numpy as np
import pandas as pd

data = np.load("synthetic_time_series.npz")

dataset = []

for key in data.files:
    y = data[key].astype(float)

    # create time index
    time_idx = np.arange(len(y))

    if len(time_idx) > 1:
        time_idx = time_idx / (len(y) - 1)
    else:
        time_idx = np.zeros_like(time_idx)

    dataset.append({
        "start": pd.Timestamp("2020-01-01"),
        "target": y,
        "feat_dynamic_real": time_idx.reshape(1, -1),  # clean shape
    })
