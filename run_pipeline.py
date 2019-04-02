"""
Top level comment: be sure to include the purpose/contents of this file
as well as the author(s)
Authors: Kenny Gwon, Raymond Liu
Purpose:
"""

import optparse
import sys

from sklearn.datasets import fetch_mldata, load_breast_cancer, fetch_20newsgroups_vectorized
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def main():

    opts = parse_args()

    if opts.dataset_name == "cancer":
        data = load_breast_cancer()
        X = data['data']
        y = data['target']
    elif opts.dataset_name == "mnist":
        data = fetch_mldata('MNIST Original', data_home="/home/smathieson/public/cs66/sklearn-data/")
        X = data['data']
        y = data['target']
        X,y = shuffle(X,y)
        X = X[:1000]
        y = y[:1000]
        X = X/255
        print(X.shape)
    elif opts.dataset_name == "news":
        data = fetch_20newsgroups_vectorized(subset='all', data_home="/home/smathieson/public/cs66/sklearn"
            "-data/")
        X = data['data']
        y = data['target']
        X,y = shuffle(X,y)
        X = X[:1000]
        y = y[:1000]
        print (X.shape)
    else:
        print("Dataset not found")
        sys.exit()

    #run this two times, once for SVM and once for Random Forest
    #SVM learner
    learner = SVC()
    parameters = {'C': [1.0, 10.0, 100.0,1000.0], 'gamma': [0.0001, 0.001, 0.01, 0.1,1]}
    testResults = runTuneTest(learner, parameters, X, y)

    print("------------------------\nSupport Vector Machine\n------------------------")
    print("Fold, Test Accuracy")
    for i in range(len(testResults)):
      print("%d, %f" % (i, testResults[i]))

    #Random Forest learner
    learner = RandomForestClassifier()
    parameters = {'n_estimators':[200], 'max_features': [0.1,0.1,0.5,1.0, "sqrt"]}
    testResults = runTuneTest(learner, parameters, X, y)

    print("--------------------\nRandom Forest\n--------------------")
    print("Fold, Test Accuracy")
    for i in range(len(testResults)):
      print("%d, %f" % (i, testResults[i]))

def parse_args():
    parser = optparse.OptionParser(description='dataset name')

    parser.add_option('-d', '--dataset_name', type='string', help='dataset name')

    (opts, args) = parser.parse_args()

    if not opts.dataset_name:
        print("Please put in a dataset name")
        parser.print_help()
        sys.exit()

    return opts

def runTuneTest(learner, parameters, X,y):
    """
    Parameters:
        Learner - the base learner
        parameters - tunable hyper-Parameters
        X - input
        y - outputs
    Purpose: This method will handle creating train/tune/test sets
    Return:
    """
    accuracyScores = []
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    for trainIndex, testIndex in skf.split(X,y):
        Xtrain, XTest = X[trainIndex], X[testIndex]
        yTrain, yTest = y[trainIndex], y[testIndex]
        clf = GridSearchCV(learner, parameters)
        clf.fit(Xtrain, yTrain)
        clf.predict(XTest)
        score = clf.score(XTest, yTest)
        accuracyScores.append(score)

    return accuracyScores

if __name__ == "__main__":
    main()
