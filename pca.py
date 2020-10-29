from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import util

def apply_pca(x, y, n_components, output_path):
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    pca = PCA(n_components)
    pca.fit(x)
    x = pca.transform(x)
    data = pd.concat([pd.DataFrame(x), y], axis=1)
    data.to_csv(output_path+str(n_components)+"_validation.csv", index=False, header=False)


if __name__ == '__main__':
    x, y = util.separe_data(r"/home/arnaldo/Documentos/hockey-fight-dataset-separada/output_amvgg19_interval_4/csv/validation.csv", 500)
    apply_pca(x, y, .50, r"/home/arnaldo/Documentos/hockey-fight-dataset-separada/")
