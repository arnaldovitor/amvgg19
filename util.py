import pandas as pd


def separe_data(input_path, n_features):
    df = pd.read_csv(input_path, header=None)
    data = pd.DataFrame(df)
    x = data.iloc[:, 0:n_features]
    y = data[n_features]
    return x, y