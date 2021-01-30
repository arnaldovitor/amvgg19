from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import util
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import os
import math
from sklearn.metrics import confusion_matrix


def apply_svm(x_train, y_train, x_test, y_test, kernel):
    clf = SVC(kernel=kernel,  probability=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    #pred_proba = clf.predict_proba(x_test)
    print("\n # SVM accuracy:", (metrics.accuracy_score(y_test, y_pred))*100)

    #clf calibrados, prefit = modelo já ajustado
    clf_calibrated = CalibratedClassifierCV(clf, cv='prefit', method='sigmoid')
    clf_calibrated.fit(x_train, y_train)

    return clf, clf_calibrated, y_pred

def apply_mlp(x_train, y_train, x_test, y_test):
    clf = MLPClassifier(random_state=1, max_iter=300)
    clf.fit(x_train, y_train)
    print("\n # MLP accuracy:", clf.score(x_test, y_test))

def apply_comb(clf_handcrafted, clf_learned, x_test_handcrafted, x_test_learned, y_test):
    all_a = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    all_combinations = []

    for a in all_a:
        y_pred_comb = []
        for i in range(len(x_test)):
            # o primeiro valor do índice corresponde ao valor do índice de y_test
            pred_proba_handcrafted = clf_handcrafted.predict_proba(x_test_handcrafted[i:i+1])
            pred_proba_learned = clf_learned.predict_proba(x_test_learned[i:i+1])
            combination = (a*pred_proba_learned[0][1])+((1-a)*pred_proba_handcrafted[0][1])

            if combination >= 0.5:
                y_pred_comb.append(1)
            else:
                y_pred_comb.append(0)

        acc = metrics.accuracy_score(y_test, y_pred_comb)*100
        all_combinations.append(acc)

    return all_combinations


def gen_venn(y_pred_handcrafted, y_pred_learned, y_test):
    handcrafted = 0
    learned = 0
    both = 0
    error = 0
    learned_i = []
    handcrafted_i = []
    both_i = []
    error_i = []

    for i in range(len(y_test)):
        if y_pred_learned[i] == y_test[i] and y_pred_handcrafted[i] == y_test[i]:
            both+=1
            both_i.append(i)
        elif y_pred_learned[i] == y_test[i] and y_pred_handcrafted[i] != y_test[i]:
            learned+=1
            learned_i.append(i)
        elif y_pred_learned[i] != y_test[i] and y_pred_handcrafted[i] == y_test[i]:
            handcrafted+=1
            handcrafted_i.append(i)
        else:
            error+=1
            error_i.append(i)

    return handcrafted, learned, both, error, handcrafted_i, learned_i, both_i, error_i


def get_videos_name(class_1_path, class_0_path, index_list):
    listing_1 = os.listdir(class_1_path)
    listing_0 = os.listdir(class_0_path)
    listing = listing_1[math.floor(len(listing_1)/2)+1:] + listing_0[math.floor(len(listing_0)/2)+1:]

    for i in range(len(listing)):
        if i in index_list:
            print(listing[i])


if __name__ == '__main__':

    #x_train, y_train = util.separe_data(r"/home/arnaldo/Documentos/violent-flows-dataset-separada/output_amvgg19_interval_4/csv/train.csv", 500)
    #x_test, y_test = util.separe_data(r"/home/arnaldo/Documentos/violent-flows-dataset-separada/output_amvgg19_interval_4/csv/test.csv", 500)
    #clf_learned, clf_learned_calibrated, y_pred_learned = apply_svm(x_train, y_train, x_test, y_test, 'rbf')
    #x_test_learned = x_test
    '''
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_learned).ravel()
    print('# learned fpr:', (fp/(fp+tn))*100)
    print('# learned fnr:', (fn/(fn+tp))*100)
    '''

    x_train, y_train = util.separe_data(r"/home/arnaldo/Documentos/cctv-fight-dataset-separada/output_mosift/csv/train.csv", 500)
    x_test, y_test = util.separe_data(r"/home/arnaldo/Documentos/cctv-fight-dataset-separada/output_mosift/csv/test.csv", 500)
    clf_handcrafted, clf_handcrafted_calibrated, y_pred_handcrafted = apply_svm(x_train, y_train, x_test, y_test, 'rbf')
    x_test_handcrafted = x_test

    #all_cc = apply_comb(clf_handcrafted, clf_learned, x_test_handcrafted, x_test_learned, y_test)
    #print(all_cc)

    '''
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_handcrafted).ravel()
    print('# handcrafted fpr:', (fp/(fp+tn))*100)
    print('# handcrafted fnr:', (fn/(fn+tp))*100)

    
    handcrafted, learned, both, error, handcrafted_i, learned_i, both_i, error_i = gen_venn(y_pred_handcrafted, y_pred_learned, np.array(y_test))

    print('\n # handcrafted:', handcrafted)
    print(handcrafted_i)
    print('\n # learned:', learned)
    print(learned_i)
    print('\n # both:', both)
    print(both_i)
    print('\n # error:', error)
    print(error_i)

    
    print('\n # learned videos name')
    get_videos_name(r"/home/arnaldo/Documentos/violent-flows-dataset-separada/output_mosift/csv/assault/", r"/home/arnaldo/Documentos/violent-flows-dataset-separada/output_mosift/csv/non-assault/", learned_i)

    print('\n # handcrafted videos name')
    get_videos_name(r"/home/arnaldo/Documentos/violent-flows-dataset-separada/output_mosift/csv/assault/", r"/home/arnaldo/Documentos/violent-flows-dataset-separada/output_mosift/csv/non-assault/", handcrafted_i)

    print('\n # both videos name')
    get_videos_name(r"/home/arnaldo/Documentos/violent-flows-dataset-separada/output_mosift/csv/assault/", r"/home/arnaldo/Documentos/violent-flows-dataset-separada/output_mosift/csv/non-assault/", both_i)

    print('\n # error videos name')
    get_videos_name(r"/home/arnaldo/Documentos/violent-flows-dataset-separada/output_mosift/csv/assault/", r"/home/arnaldo/Documentos/violent-flows-dataset-separada/output_mosift/csv/non-assault/", error_i)
    '''