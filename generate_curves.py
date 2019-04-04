"""
Top level comment: be sure to include the purpose/contents of this file
as well as the author(s)
Purpose: The purpose of this fie is to generate learning curves for
         random forest and suport vector machine classifiers. We will vary
         one of the hyper-parameters and see how the train and test error 
         accuracies change. 
Authors: Raymond Liu and Kenny Gwon
"""

import optparse
import numpy as np
import run_pipeline
import sys
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata, load_breast_cancer, fetch_20newsgroups_vectorized
from sklearn.utils import shuffle
from sklearn.model_selection import validation_curve
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
    elif opts.dataset_name == "news":
        data = fetch_20newsgroups_vectorized(subset='all', data_home="/home/smathieson/public/cs66/sklearn"
            "-data/")
        X = data['data']
        y = data['target']
        X,y = shuffle(X,y)
        X = X[:1000]
        y = y[:1000]
    else:
        print("Dataset not found")
        sys.exit()

    #SVM learner
    learner = SVC()
    #creates a list of gamma value ranges
    gammaRange = np.logspace(-5,1,7)
    
    trainScores, testScores = validation_curve(learner, X, y,'gamma', gammaRange)

    avgTrainAcc = []
    avgTestAcc = []
    print("Gamma Value, Train Accuracy, Test Accuracy")
    for i in range(len(testScores)):
      avgAccuracyList = avgAccuracy(trainScores[i], testScores[i])
      avgTrainAcc.append(avgAccuracyList[0])
      avgTestAcc.append(avgAccuracyList[1])
      print("%7.2g %f %f" % (gammaRange[i], avgAccuracyList[0], avgAccuracyList[1]))
    
    #plots
    fig = plt.figure()
    plt.plot(gammaRange, avgTrainAcc, 'bo-')
    plt.plot(gammaRange, avgTestAcc, 'r*-')
    plt.xscale('log')
    plt.xlabel("Gamma Range")
    plt.ylabel("Accuracy")
    plt.title("MNIST Dataset - Accuracy vs Gamma Range")
    plt.legend(["Train Accuracy", "Test Accuracy"])
    plt.show()

    #Random Forest learner
    learner = RandomForestClassifier()
    #creates a list of n_estimator values that the parameter "n_estimator" will take on
    estimatorRange = list(range(1,202,10))

    trainScores, testScores = validation_curve(learner, X, y, 'n_estimators', estimatorRange)
    
    avgTrainAcc = []
    avgTestAcc = []
    print("\nNumber of Trees, Train Accuracy, Test Accuracy")
    for i in range(len(testScores)):
      avgAccuracyList = avgAccuracy(trainScores[i], testScores[i])
      avgTrainAcc.append(avgAccuracyList[0])
      avgTestAcc.append(avgAccuracyList[1])
      print("%6.1d %f %f" % (estimatorRange[i], avgAccuracyList[0], avgAccuracyList[1]))

    #plots
    fig = plt.figure()
    plt.plot(estimatorRange, avgTrainAcc, 'bo-')
    plt.plot(estimatorRange, avgTestAcc, 'r*-')
    plt.xlabel("Number of Estimators")
    plt.ylabel("Accuracy")
    plt.title("MNIST Dataset - Accuracy vs Number of Estimators")
    plt.legend(["Train Accuracy", "Test Accuracy"])
    plt.show() 


def parse_args():
    parser = optparse.OptionParser(description='dataset name')

    parser.add_option('-d', '--dataset_name', type='string', help='dataset name')

    (opts, args) = parser.parse_args()

    if not opts.dataset_name:
        print("Please put in a dataset name")
        parser.print_help()
        sys.exit()

    return opts

def avgAccuracy(trainRow, testRow):
  """
  parameters: trainRow - one row of the training accuracy values
              testRow - one row of the test accuracy values
  return: avgAccuracyList - List of the average accuracy values. First index
          represents the average train accuracy and the second index represents
          the average test accuracy
  """
  avgAccuracyList =[]

  trainSum = sum(trainRow)
  trainAvg = trainSum/len(trainRow)

  testSum = sum(testRow)
  testAvg = testSum/len(testRow)

  avgAccuracyList.append(trainAvg)
  avgAccuracyList.append(testAvg)

  return avgAccuracyList


if __name__ == "__main__":
    main()
