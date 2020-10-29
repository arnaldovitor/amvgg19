from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import util
from sklearn import metrics
import numpy as np


def apply_svm(x_train, y_train, x_test, y_test, kernel):
    clf = SVC(kernel=kernel)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("# SVM accuracy:", metrics.accuracy_score(y_test, y_pred))
    return y_pred


def apply_mlp(x_train, y_train, x_test, y_test):
    clf = MLPClassifier(random_state=1, max_iter=300)
    clf.fit(x_train, y_train)
    print("# MLP accuracy:", clf.score(x_test, y_test))


def gen_venn(y_pred_handcrafted, y_pred_learned, y_test):
    handcrafted = 0
    learned = 0
    both = 0
    error = 0

    for i in range(len(y_test)):
        if y_pred_learned[i] == y_test[i] and y_pred_handcrafted[i] == y_test[i]:
            both+=1
        elif y_pred_learned[i] == y_test[i] and y_pred_handcrafted[i] != y_test[i]:
            learned+=1
        elif y_pred_learned[i] != y_test[i] and y_pred_handcrafted[i] == y_test[i]:
            handcrafted+=1
        else:
            error+=1

    return handcrafted, learned, both, error


if __name__ == '__main__':
    x_train, y_train = util.separe_data(r"/home/arnaldo/Documentos/hockey-fight-dataset-separada/output_amvgg19_interval_4/csv/train.csv", 500)
    x_test, y_test = util.separe_data(r"/home/arnaldo/Documentos/hockey-fight-dataset-separada/output_amvgg19_interval_4/csv/test.csv", 500)
    y_pred_learned = apply_svm(x_train, y_train, x_test, y_test, 'rbf')

    x_train, y_train = util.separe_data(r"/home/arnaldo/Documentos/hockey-fight-dataset-separada/output_mosift/csv/train.csv", 500)
    x_test, y_test = util.separe_data(r"/home/arnaldo/Documentos/hockey-fight-dataset-separada/output_mosift/csv/test.csv", 500)
    y_pred_handcrafted = apply_svm(x_train, y_train, x_test, y_test, 'rbf')

    handcrafted, learned, both, error = gen_venn(y_pred_handcrafted, y_pred_learned, np.array(y_test))

    print('# handcrafted:', handcrafted)
    print('# learned:', learned)
    print('# both:', both)
    print('# error:', error)