'''
Paper Title: Road Surface Recognition Based on DeepSense Neural Network using Accelerometer Data
Created by ITS Lab, Institute of Computer Science, University of Tartu
'''

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import utils


def RandomForest(train_data, train_label, test_data, test_label,
                 n_estimator='warn', random_state=53):

    rf = RandomForestClassifier(n_estimators=n_estimator, random_state=random_state)
    rf.fit(train_data.values, train_label)
    print('Feature importances: ', rf.feature_importances_)
    pred = rf.predict(test_data.values)
    acc = utils.evaluate_accuracy(pred, test_label, p=True)

    # evaluate each category
    cat1, cat2, cat3 = [], [], []
    cat = [cat1, cat2, cat3]
    for i, l in enumerate(test_label):
        cat[l].append(i)
    for i, c in enumerate(cat):
        ctest, clabels = test_data.values[c], test_label[c]
        pred = rf.predict(ctest)
        acc = utils.evaluate_accuracy(pred, clabels, p=False)
        print('Category %d, accuracy: %.4f' % (i, acc))
    return rf


def SVM(train_data, train_label, test_data, test_label,
        kernel='rbf', gamma='scale', C=1.0, degree=3, max_iter=-1):

    clf = svm.SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, max_iter=max_iter,
                  decision_function_shape='ovr')

    clf.fit(train_data.values, train_label)
    pred = clf.predict(test_data.values)
    acc = utils.evaluate_accuracy(pred, test_label, p=True)

    # evaluate each category
    cat1, cat2, cat3 = [], [], []
    cat = [cat1, cat2, cat3]
    for i, l in enumerate(test_label):
        cat[l].append(i)
    for i, c in enumerate(cat):
        ctest, clabels = test_data.values[c], test_label[c]
        pred = clf.predict(ctest)
        acc = utils.evaluate_accuracy(pred, clabels, p=False)
        print('Category %d, accuracy: %.4f' % (i, acc))
    return clf
