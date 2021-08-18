from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import util
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import os
import math
import tensorflow as tf
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

def apply_tree(x_train, y_train, x_test, y_test):
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("\n # Decision Tree accuracy:", metrics.accuracy_score(y_test, y_pred))
    return y_pred

def apply_mlp(x_train, y_train, x_test, y_test):
    clf = MLPClassifier(random_state=1, max_iter=300)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    #print("\n # MLP accuracy:", clf.score(x_test, y_test))
    return y_pred

def apply_fcn(x_train, y_train, x_test, y_test):
    class Predictor(tf.keras.Model):
        def __init__(self):
            super(Predictor, self).__init__()
            self.dense0 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
            self.dense1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
            self.dense2 = tf.keras.layers.Dense(16, activation=tf.nn.relu)
            self.dense3 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
            self.dropout_layer = tf.keras.layers.Dropout(rate=0.1)

        def call(self, inputs):
            x = self.dense0(inputs)
            x = self.dropout_layer(x)
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dense3(x)
            return x

    model = Predictor()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                     loss=tf.keras.losses.BinaryCrossentropy(),
                     metrics=[tf.keras.metrics.BinaryAccuracy(), ])

    history = model.fit(x_train, y_train, epochs=800, validation_data=(x_test, y_test))
    model.save('model_path/', save_format='tf')
    y_pred = model.predict(x_test)

    return y_pred

def apply_comb(clf_handcrafted, clf_learned, x_test_handcrafted, x_test_learned, y_test):
    all_a = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    all_combinations = []

    for a in all_a:
        y_pred_comb = []
        for i in range(len(x_test_learned)):
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

def normalize_x(x):
  coef = np.max(x, axis=0)
  coef[coef == 0] = np.max(coef)
  x = x/coef[None, :]
  np.save("maximus.npy", coef)

  return x

def normalize_x_test(x):
    coef = np.load("maximus.npy")
    x = x/coef[None, :]
    return x


if __name__ == '__main__':
    X, y = util.separe_data(r"train.csv", 128)
    X = normalize_x(X)

    X_val, y_val = util.separe_data(r"val.csv", 128)
    X_val = normalize_x_test(X_val)

    y_pred = apply_fcn(X, y, X_val, y_val)
