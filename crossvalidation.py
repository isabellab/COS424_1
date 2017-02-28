# Written by Isabella Bosetti
# For COS 424 SPRING 2017
# based on example methods seen at http://scikit-learn.org/stable/
# used with gratitude and with no intention of infringement

# Performs cross-validation
# change filepaths on lines 25 through 32


import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.ensemble import RandomForestClassifier as RForest
from sklearn.tree import DecisionTreeClassifier as DTree
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit

from sklearn import svm
import pandas as pd
from ggplot import *



trainData = np.genfromtxt('bow1/out_bag_of_words_5.csv', delimiter=',')     #xtrain
trainOutcomes = np.loadtxt('bow1/out_classes_5.txt')                        #ytrain

testData = np.genfromtxt('running/out_bag_of_words_0.csv', delimiter=',')   #xtest
testOutcomes = np.loadtxt('running/out_classes_0.txt')         #ytest

xData = np.array(np.concatenate((trainData, testData)))
yTarget = np.array(np.concatenate((trainOutcomes, testOutcomes)))

print xData.shape
print yTarget.shape

print "LogisticRegression:"
logreg = LogReg()
scores = cross_val_score(logreg, xData, yTarget, cv = 10)
scores

print ("Accuracy: %0.2f (+/- %.02f)" % (scores.mean(), scores.std() *2))


print "BernoulliNB:"
bernoulli = BernoulliNB()

scores = cross_val_score(bernoulli, xData, yTarget, cv = 20)
scores

print ("Accuracy: %0.2f (+/- %.02f)" % (scores.mean(), scores.std() *2))


print "LinearSVC:"
linsvc = LinearSVC()

scores = cross_val_score(linsvc, xData, yTarget, cv = 10)
scores

print ("Accuracy: %0.2f (+/- %.02f)" % (scores.mean(), scores.std() *2))


print "RandomForestClassifier"
rforest = RForest()

scores = cross_val_score(rforest, xData, yTarget, cv = 10)
scores

print ("Accuracy: %0.2f (+/- %.02f)" % (scores.mean(), scores.std() *2))

print "DecisionTreeClassifier"
dtree = DTree()

scores = cross_val_score(dtree, xData, yTarget, cv = 10)
scores

print ("Accuracy: %0.2f (+/- %.02f)" % (scores.mean(), scores.std() *2))
