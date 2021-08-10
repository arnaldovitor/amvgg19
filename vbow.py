from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import os
import pickle

def clustering(dict, n_clusters, batch_size):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size).fit(dict)
    return kmeans


def gen_histogram(feature_vectors, kmeans):
    histogram = np.zeros(len(kmeans.cluster_centers_))
    cluster_result = kmeans.predict(feature_vectors)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram


def gen_vbow(input_path, output_name, kmeans, class_id):
    vbow = []
    listing = os.listdir(input_path)
    listing.sort()
    for csv_name in listing:
        try:
            feature_vectors = pd.read_csv(input_path+csv_name)
            histogram = gen_histogram(feature_vectors, kmeans)
            histogram = np.append(histogram, class_id)
            vbow.append(histogram)
        except:
            histogram = np.zeros(1024)
            histogram = np.append(histogram, class_id)
            vbow.append(histogram)

    np.savetxt(output_name+".csv", vbow, delimiter=",")


if __name__ == '__main__':
    dict = pd.read_csv(r"/home/arnaldo/Documents/violent-flows-dataset-separada/csv/dict.csv")
    kmeans = clustering(dict, 256, 32)
    pickle.dump(kmeans, open("clt.pkl", "wb"))
    print("# clustering ended")
    gen_vbow(r"/home/arnaldo/Documents/violent-flows-dataset-separada/csv/train/violence/", r"/home/arnaldo/Documents/violent-flows-dataset-separada/csv/train_violence_1024", kmeans, 1)
    gen_vbow(r"/home/arnaldo/Documents/violent-flows-dataset-separada/csv/train/non-violence/", r"/home/arnaldo/Documents/violent-flows-dataset-separada/csv/train_non_violence_1024", kmeans, 0)
    gen_vbow(r"/home/arnaldo/Documents/violent-flows-dataset-separada/csv/test/violence/", r"/home/arnaldo/Documents/violent-flows-dataset-separada/csv/test_violence_1024", kmeans, 1)
    gen_vbow(r"/home/arnaldo/Documents/violent-flows-dataset-separada/csv/test/non-violence/", r"/home/arnaldo/Documents/violent-flows-dataset-separada/csv/test_non_violence_1024", kmeans, 0)
