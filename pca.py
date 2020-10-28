from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


def separe_data(input_path, n_features):
    df = pd.read_csv(input_path, header=None)
    data = pd.DataFrame(df)
    print(data)
    X = data.iloc[:, 0:n_features]
    Y = data[n_features]
    return X, Y


def apply_pca(X, Y, n_components, output_path):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    pca = PCA(n_components)
    pca.fit(X)
    X = pca.transform(X)
    data = pd.concat([pd.DataFrame(X), Y], axis=1)
    data.to_csv(output_path+str(n_components)+"_c_n_validation.csv", index=False, header=False)


if __name__ == '__main__':
    X, Y = separe_data(r"/home/arnaldo/Documentos/hockey-fight-dataset-separada/output_amvgg19_interval_4/csv/validation.csv", 500)
    apply_pca(X, Y, .50, r"/home/arnaldo/Documentos/hockey-fight-dataset-separada/")
