# Written by Isabella Bosetti and Matthew Penza
# For COS 424 SPRING 2017
# based on example methods seen at http://scikit-learn.org/stable/
# used with gratitude and with no intention of infringement

# runs models on bag of words data representation
# will work equally as well on bigram bag of words
# replace file paths on lines 22 through 26

import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.ensemble import RandomForestClassifier as RForest
from sklearn.tree import DecisionTreeClassifier as DTree
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score , recall_score, f1_score
import pandas as pd
from ggplot import *
import timeit


trainData = np.genfromtxt('bow1/out_bag_of_words_5.csv', delimiter=',')     #xtrain
trainOutcomes = np.loadtxt('bow1/out_classes_5.txt')                        #ytrain

testData = np.genfromtxt('running/out_bag_of_words_0.csv', delimiter=',')   #xtest
testOutcomes = np.loadtxt('running/out_classes_0.txt')         #ytest

def predictions(classifier):
    if (type(classifier) is BernoulliNB) or (type(classifier) is DTree) or (type(classifier) is RForest):
        preds = classifier.predict_proba(testData)[:,1]
    elif (type(classifier) is LinearSVC) or (type(classifier) is LogReg):
        preds = classifier.decision_function(testData)
    fpr, tpr, _ = roc_curve(testOutcomes, preds)
    # print fpr
    # print tpr
    print auc(fpr, tpr)
    # print ''
    df = pd.DataFrame(dict(fpr=fpr, tpr = tpr))
    scores = classifier.score(testData, testOutcomes)
    print ("scores: %0.2f " % (scores))
    # print tpr.shape
    # print ("true positive rate is: %0.2f " % )
    # print ("false positive rate is: %0.2f " % fpr.value)
    prec = precision_score(testOutcomes, classifier.predict(testData))
    print  ("precision: %0.2f " % (prec))
    rec = recall_score(testOutcomes, classifier.predict(testData))
    print  ("recall: %0.2f " % (rec))
    f1 = f1_score(testOutcomes, classifier.predict(testData))
    print ("F1 score: %0.2f" % (f1))



    

print "BernoulliNB:"
t0 = timeit.default_timer()
bernoulli = BernoulliNB()
bernoulli.fit(trainData, trainOutcomes)
end = timeit.default_timer()
print "fit time: "
print  (end- t0)
predictions(bernoulli)


print "LinearSVC:"
t0 = timeit.default_timer()
linsvc = LinearSVC()
linsvc.fit(trainData, trainOutcomes)
end = timeit.default_timer()
print "fit time: "
print  (end- t0)
predictions(linsvc)


print "LogisticRegression:"
t0 = timeit.default_timer()
logreg = LogReg()
logreg.fit(trainData, trainOutcomes)
end = timeit.default_timer()
print "fit time: "
print  (end- t0)
predictions(logreg)



print "RandomForestClassifier"
t0 = timeit.default_timer()
rforest = RForest()
rforest.fit(trainData, trainOutcomes)
end = timeit.default_timer()
print "fit time: "
print  (end- t0)
predictions(rforest)



print "Training data on DecisionTreeClassifier"
t0 = timeit.default_timer()
dtree = DTree()
dtree.fit(trainData, trainOutcomes)
end = timeit.default_timer()
print "fit time: "
print  (end- t0)
predictions(dtree)




