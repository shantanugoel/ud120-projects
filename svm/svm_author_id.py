#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
#l = int(len(features_train)/100)
#features_train = features_train[:l]
#labels_train = labels_train[:l]
clf = svm.SVC(kernel='rbf', C=10000)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print(pred[10])
print(pred[26])
print(pred[50])
print(accuracy_score(labels_test, pred))
c_count = 0
s_count = 1
for p in pred:
    if p == 0:
        s_count += 1
    else:
        c_count += 1
print(c_count)
print(s_count)

#########################################################


