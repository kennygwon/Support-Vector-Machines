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
from scipy.stats import ttest_rel


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
    print("\n------------------------\nSupport Vector Machine\n------------------------")
    learner = SVC()
    parameters = {'C': [1.0, 10.0, 100.0,1000.0], 'gamma': [0.0001, 0.001, 0.01, 0.1,1]}
    testResults_SVM = runTuneTest(learner, parameters, X, y)

    print("Fold Test Accuracy")
    for i in range(len(testResults_SVM)):
      print("%4d %4f" % (i, testResults_SVM[i]))

    #Random Forest learner
    print("\n--------------------\nRandom Forest\n--------------------")
    learner = RandomForestClassifier()
    parameters = {'n_estimators':[200], 'max_features': [0.1,0.1,0.5,1.0, "sqrt"]}
    testResults_RF = runTuneTest(learner, parameters, X, y)

    print("Fold Test Accuracy")
    for i in range(len(testResults_RF)):
      print("%4d %4f" % (i, testResults_RF[i]))

    p_value = ttest_rel(testResults_RF, testResults_SVM)
    print("\np_value =", p_value.pvalue)

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
    Return: returns a list of accuracies
    """
    #list of accuracy scores
    accuracyScores = []
    #counter used for printing purposes
    counter = 1 
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    for trainIndex, testIndex in skf.split(X,y):
        Xtrain, XTest = X[trainIndex], X[testIndex]
        yTrain, yTest = y[trainIndex], y[testIndex]
        clf = GridSearchCV(learner, parameters)
        clf.fit(Xtrain, yTrain)
        clf.predict(XTest)
        score = clf.score(XTest, yTest)
        scoreTrain = clf.score(Xtrain, yTrain)
        print("Fold %d" % (counter))
        print("Best parameter:", clf.best_params_)
        print("Training score %f\n" % (scoreTrain))
        counter += 1
        accuracyScores.append(score)

    return accuracyScores

if __name__ == "__main__":
    main()
