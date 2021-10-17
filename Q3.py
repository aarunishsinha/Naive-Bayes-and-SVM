import os
import sys
import time
import re
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

def convert(filename):
    f = open(filename, 'r')
    string = f.read().strip()[1:]
    string = string[:-1]
    preds = string.split(", ")
    preds = np.array(preds).astype(int)
    return preds

def initialise(filename):
    X = []
    Y = []
    f = open(filename, mode = 'r')
    for review in f:
        content = json.loads(review)
        text = content["text"].strip()
        text_in_lower = text.lower()
        p_text = re.sub(r'[^\w\s]','',text_in_lower)
        p_text = re.sub(r'[^\w\s]','',p_text)
        p_text = re.sub('\r?\n',' ',p_text)
        X.append(p_text)
        Y.append(int(content["stars"]))
    return X,Y
def liblinear_svm(train_filename,test_filename,output_filename):
    trainX,trainY = initialise(train_filename)
    testX,testY = initialise(test_filename)
    trainX,valX,trainY,valY = train_test_split(trainX,trainY,test_size=0.1)
    Cs = [10, 5, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    max_c = 0
    max_acc = 0
    for c in Cs:
        # print ("In for loop, c = ",c)
        text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', LinearSVC(C=c))])
        # print ("Start fitting")
        text_clf.fit(trainX,trainY)
        # print ("Predicting")
        predicted = text_clf.predict(valX)
        # print ("Scoring")
        con = confusion_matrix(valY,predicted)
        acc = np.trace(con)/(np.shape(valY)[0])
        # print ("For "+str(c)+" accuracy is: "+str(acc))
        if acc>max_acc:
            max_acc = acc
            max_c = c
    start = time.time()
    # print (max_c)
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', LinearSVC(C=max_c))])
    text_clf.fit(trainX,trainY)
    end = time.time()
    print("Training Time:", (end-start))
    predicted = text_clf.predict(testX)
    test_con = confusion_matrix(testY,predicted)
    print (test_con)
    test_acc = np.trace(test_con)/(np.shape(testY)[0])
    print (test_acc)
def SGD_SVM(train_filename,test_filename,output_filename):
    trainX,trainY = initialise(train_filename)
    testX,testY = initialise(test_filename)
    output = open(output_filename,'w')
    alphas = [10, 5, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    trainX,valX,trainY,valY = train_test_split(trainX,trainY,test_size=0.1)
    max_acc = 0
    max_alpha = 0
    for a in alphas:
        clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='log', penalty='l2',alpha=a))])
        clf.fit(trainX,trainY)
        predicted = clf.predict(valX)
        con = confusion_matrix(valY,predicted)
        acc = np.trace(con)/(np.shape(valY)[0])
        if acc>max_acc:
            max_acc = acc
            max_alpha = a

    start = time.time()
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='log', penalty='l2',alpha=max_alpha))])
    text_clf.fit(trainX, trainY)
    end = time.time()
    # print("Training Time:", (end-start))
    predicted = text_clf.predict(testX)
    output.write(str(list(predicted)))
    output.close()
    predicted_read = convert(output_filename)
    test_con = confusion_matrix(testY,predicted)
    test_con_read = confusion_matrix(testY,predicted_read)
    # print (test_con)
    test_acc = np.trace(test_con)/(np.shape(testY)[0])
    test_acc_read = np.trace(test_con_read)/(np.shape(testY)[0])
    print (test_acc)
    print ("read acc:", test_acc_read)
def main():
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    output_filename = sys.argv[3]
    SGD_SVM(train_filename,test_filename,output_filename)
    # liblinear_svm(train_filename,test_filename,output_filename)
if __name__ == "__main__":
    main()
