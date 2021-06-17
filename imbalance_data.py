import os
import numpy as np
from collections import Counter
import sklearn

data_path = "./data/raw/NATOPS"
imbalance_data_path = "./data/raw/NATOPS_IMBALANCE"

X_train = np.load(os.path.join(data_path, "X_train.npy"))
X_test = np.load(os.path.join(data_path, "X_test.npy"))
y_train = np.load(os.path.join(data_path, "y_train.npy"))
y_test = np.load(os.path.join(data_path, "y_test.npy"))

print(Counter(y_train[:, 0]))
print(Counter(y_test[:, 0]))


def sortout_class(X, y, label=0):
    idx = np.where(y[:, 0] == label)
    y_class = y[idx, :]
    X_class = X[idx, :, :]
    return X_class[0], y_class[0]

POS_NUM = 5

def create_dataset():
    X_train_class_0, y_train_class_0 = sortout_class(X_train, y_train, label=0)
    X_test_class_0, y_test_class_0 = sortout_class(X_train, y_train, label=0)
    X_train_class_1, y_train_class_1 = sortout_class(X_test, y_test, label=1)
    X_test_class_1, y_test_class_1 = sortout_class(X_test, y_test, label=1)

    # imbalance: 30:3
    X_train_class_1, y_train_class_1 = X_train_class_1[:POS_NUM, :, :], y_train_class_1[:POS_NUM, :]
    X_test_class_1, y_test_class_1 = X_test_class_1[:POS_NUM, :, :], y_test_class_1[:POS_NUM, :]

    X_train_new = np.vstack((X_train_class_0, X_train_class_1))
    y_train_new = np.vstack((y_train_class_0, y_train_class_1))
    X_train_new, y_train_new = sklearn.utils.shuffle(X_train_new, y_train_new)

    X_test_new = np.vstack((X_test_class_0, X_test_class_1))
    y_test_new = np.vstack((y_test_class_0, y_test_class_1))
    X_test_new, y_test_new = sklearn.utils.shuffle(X_test_new, y_test_new)

    if not os.path.exists(imbalance_data_path):
        os.mkdir(imbalance_data_path)

    np.save(os.path.join(imbalance_data_path, "X_train.npy"), X_train_new)
    np.save(os.path.join(imbalance_data_path, "X_test.npy"), X_test_new)
    np.save(os.path.join(imbalance_data_path, "y_train.npy"), y_train_new)
    np.save(os.path.join(imbalance_data_path, "y_test.npy"), y_test_new)

create_dataset()
