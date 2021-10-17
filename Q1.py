import sys
import os
import re
import json
import time
import math
import numpy as np
import pandas as pd
import nltk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import trigrams
from nltk.util import ngrams
from nltk import bigrams
import matplotlib.pyplot as plt

def convert(filename):
    f = open(filename, 'r')
    string = f.read().strip()[1:]
    string = string[:-1]
    preds = string.split(", ")
    preds = np.array(preds).astype(int)
    return preds

# Stemming Functions copied from utils.py
def _stem(doc, p_stemmer, en_stop, return_tokens):
    tokens = word_tokenize(doc.lower())
    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stemmed_tokens = map(lambda token: p_stemmer.stem(token), stopped_tokens)
    if not return_tokens:
        return ' '.join(stemmed_tokens)
    return list(stemmed_tokens)
def getStemmedDocuments(docs, return_tokens=True):
    """
        Args:
            docs: str/list(str): document or list of documents that need to be processed
            return_tokens: bool: return a re-joined string or tokens
        Returns:
            str/list(str): processed document or list of processed documents
        Example:
            new_text = "It is important to by very pythonly while you are pythoning with python.
                All pythoners have pythoned poorly at least once."
            print(getStemmedDocuments(new_text))
        Reference: https://pythonprogramming.net/stemming-nltk-tutorial/
    """
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    if isinstance(docs, list):
        output_docs = []
        for item in docs:
            output_docs.append(_stem(item, p_stemmer, en_stop, return_tokens))
        return output_docs
    else:
        return _stem(docs, p_stemmer, en_stop, return_tokens)

# Read datasets
def initialise(filename):
    X = {}
    Y = {}
    i = 0
    f = open(filename, mode = 'r')
    for review in f:
        content = json.loads(review)
        text = content["text"].strip()
        text_in_lower = text.lower()
        p_text = re.sub(r'[^\w\s]','',text_in_lower)
        p_text = re.sub(r'[^\w\s]','',p_text)
        p_text = re.sub('\r?\n',' ',p_text)
        X[i] = p_text
        Y[i] = int(content["stars"])
        i += 1
    return X,Y
def roc_calc_plot(testY, predict_value, prediction):
    test = np.zeros(len(testY))
    for i in range(len(testY)):
        if testY[i] == prediction[i]:
            test[i] = 1
    fpr = dict()
    tpr = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(test, predict_value)
    plt.figure()
    plt.plot(fpr[1],tpr[1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig("roc_stemming.png", dpi=200)
    plt.show()
# Processing on Raw Words
def split_into_words(trainX, trainY):
    dict = {}
    # class_occur stores the number of times each class has occured
    class_occur = [0,0,0,0,0]
    # class_vocab stores the number of words in each of the classes
    class_vocab = [0,0,0,0,0]
    tokenizer = RegexpTokenizer("[\w']+")
    for i in range(len(trainX)):
        stars = trainY[i]
        # words = trainX[i].split()
        words  = tokenizer.tokenize(trainX[i])
        class_occur[stars-1] += 1
        class_vocab[stars-1] += len(words)
        for word in words:
            if word not in dict:
                dict[word] = [0,0,0,0,0]
            dict[word][stars-1] += 1
    return dict,class_occur,class_vocab
def predict(dictionary, testX, testY, class_vocab, class_prior):
    tokenizer = RegexpTokenizer("[\w']+")
    predict_value = np.zeros(len(testX))
    prediction = np.zeros(len(testX))
    for i in range(len(testX)):
        p = np.zeros(5)
        # words = testX[i].split()
        words = tokenizer.tokenize(testX[i])
        for j in range(5):
            p[j] += class_prior[j]
            for word in words:
                key = dictionary.get(word, None)
                if key is not None:
                    p[j] += dictionary[word][j]
                else:
                    p[j] -= math.log((float(len(dictionary)+class_vocab[j])))
        prediction[i] = 1 + np.argmax(p)
        predict_value[i] = math.exp(max(p))
    # roc_calc_plot(testY, predict_value, prediction)
    # print (predict_value)
    return prediction
# Accuracy Calculation
def accuracy(Y, prediction):
    acc = 0
    for i in range(len(Y)):
        if Y[i] == prediction[i]:
            acc += 1
    accur = acc/len(Y)
    return accur

# Plotting the Confusion Matrix
def matrix_plot(confusion_mat):
    plt.imshow(confusion_mat)
    plt.title("Confusion Matrix for Test Set Prediction(Naive Bayes)")
    plt.colorbar()
    plt.set_cmap("Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("conmat.png",dpi=200)
    plt.show()

# Proceessing on Stemmed Words
def stem_words(trainX, trainY):
    dict = {}
    class_occur = [0,0,0,0,0]
    class_vocab = [0,0,0,0,0]
    for i in range(len(trainX)):
        stars = trainY[i]
        stemmed_words = getStemmedDocuments(trainX[i], True)
        class_vocab[stars-1] += len(stemmed_words)
        class_occur[stars-1] += 1
        for word in stemmed_words:
            key = dict.get(word, None)
            if key is None:
                dict[word] = [0,0,0,0,0]
            dict[word][stars-1] += 1
    return dict,class_occur,class_vocab
def predict_for_stemmed(dictionary, testX, testY, class_vocab, class_prior):
    prediction = np.zeros(len(testX))
    predict_value = np.zeros(len(testX))
    for i in range(len(testX)):
        p = np.zeros(5)
        stemmed_words = getStemmedDocuments(testX[i],True)
        for j in range(5):
            p[j] += class_prior[j]
            for word in stemmed_words:
                key = dictionary.get(word, None)
                if key is not None:
                    p[j] += dictionary[word][j]
                else:
                    p[j] -= math.log((float(len(dictionary)+class_vocab[j])))
        prediction[i] = 1 + np.argmax(p)
        predict_value[i] = math.exp(max(p))
    roc_calc_plot(testY, predict_value, prediction)
    return prediction

# Feature Engineering : BIGRAMS
def _bigram(doc, return_tokens):
    tokenizer = RegexpTokenizer("[\w']+")
    tokens = tokenizer.tokenize(doc.lower())
    # en_stop = set(stopwords.words('english'))   ******** UNCOMMENT TO REMOVE STOPWORDS *******
    # stop_tokens = filter(lambda token: token not in en_stop, tokens)
    # bigram_tokens = bigrams(tokens)    #*****UNCOMMENT TO USE BIGRAMS******
    bigram_tokens = trigrams(tokens)
    # bigram_tokens = ngrams(tokens, 4)  #*****UNCOMMENT TO USE N-GRAMS******
    if not return_tokens:
        return ' '.join(bigram_tokens)
    return list(bigram_tokens)
def getBigramDocuments(docs, return_tokens=True):
    if isinstance(docs, list):
        output_docs = []
        for item in docs:
            output_docs.append(_bigram(item, return_tokens))
        return output_docs
    else:
        return _bigram(docs, return_tokens)
def bigram_words(trainX, trainY):
    dict = {}
    class_occur = [0,0,0,0,0]
    class_vocab = [0,0,0,0,0]
    for i in range(len(trainX)):
        stars = trainY[i]
        b_words = getBigramDocuments(trainX[i], True)
        class_vocab[stars-1] += len(b_words)
        class_occur[stars-1] += 1
        for word in b_words:
            key = dict.get(word, None)
            if key is None:
                dict[word] = [0,0,0,0,0]
            dict[word][stars-1] += 1
    return dict,class_occur,class_vocab
def predict_for_bigram(dictionary, testX, testY, class_vocab, class_prior):
    prediction = np.zeros(len(testX))
    predict_value = np.zeros(len(testX))
    for i in range(len(testX)):
        p = np.zeros(5)
        b_words = getBigramDocuments(testX[i],True)
        for j in range(5):
            p[j] += class_prior[j]
            for word in b_words:
                key = dictionary.get(word, None)
                if key is not None:
                    p[j] += dictionary[word][j]
                else:
                    p[j] -= math.log((float(len(dictionary)+class_vocab[j])))
        prediction[i] = 1 + np.argmax(p)
        predict_value[i] = math.exp(max(p))
    # roc_calc_plot(testY, predict_value, prediction)
    return prediction

#Feature Engineering : LEMMATIZER
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
def _lemmatize(doc, return_tokens):
    # POS Tagging and Removing Stopwords
    tokenizer = RegexpTokenizer("[\w']+")
    tokens = tokenizer.tokenize(doc)
    # en_stop = set(stopwords.words('english'))   ******** UNCOMMENT TO REMOVE STOPWORDS *******
    # stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    tagged = nltk.pos_tag(tokens)
    # Lemmatizing
    lt = WordNetLemmatizer()
    lemma_tokens = []
    for word,tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:
            lemma = lt.lemmatize(word)
        else:
            lemma = lt.lemmatize(word, pos=wntag)
        lemma_tokens.append(lemma)
    # lemma_tokens = lt.lemmatize(doc, wordnet.ADJ)
    if not return_tokens:
        return ' '.join(lemma_tokens)
    return list(lemma_tokens)
def getLemmatizeDocuments(docs, return_tokens = True):
    if isinstance(docs, list):
        output_docs = []
        for item in docs:
            output_docs.append(_lemmatize(item, return_tokens))
        return output_docs
    else:
        return _lemmatize(docs, return_tokens)
def lemmatize_words(trainX, trainY):
    dict = {}
    class_occur = [0,0,0,0,0]
    class_vocab = [0,0,0,0,0]
    for i in range(len(trainX)):
        stars = trainY[i]
        l_words = getLemmatizeDocuments(trainX[i], True)
        class_vocab[stars-1] += len(l_words)
        class_occur[stars-1] += 1
        for word in l_words:
            key = dict.get(word, None)
            if key is None:
                dict[word] = [0,0,0,0,0]
            dict[word][stars-1] += 1
    return dict,class_occur,class_vocab
def predict_for_lemmatizer(dictionary, testX, testY, class_vocab, class_prior):
    prediction = np.zeros(len(testX))
    for i in range(len(testX)):
        p = np.zeros(5)
        l_words = getLemmatizeDocuments(testX[i],True)
        for j in range(5):
            p[j] += class_prior[j]
            for word in l_words:
                key = dictionary.get(word, None)
                if key is not None:
                    p[j] += dictionary[word][j]
                else:
                    p[j] -= math.log((float(len(dictionary)+class_vocab[j])))
        prediction[i] = 1 + np.argmax(p)
    return prediction

def main():
    train_filename = sys.argv[1]
    trainX,trainY = initialise(train_filename)
    test_filename = sys.argv[2]
    testX,testY = initialise(test_filename)
    output_filename = sys.argv[3]

    # **************** UNCOMMENT TO RUN THE MODEL ON RAW WORDS *****************
    # dict,class_occur,class_vocab = split_into_words(trainX,trainY)
    # # Log Likelihood Calculation
    # dictionary = dict
    # for word in dictionary:
    #     for i in range(5):
    #         dictionary[word][i] = math.log((float(dictionary[word][i])+ 1)/(float(len(dictionary) + class_vocab[i])))
    # # Log Prior Calculation
    # class_prior = [0.0,0.0,0.0,0.0,0.0]
    # for i in range(5):
    #     class_prior[i] = math.log((float(class_occur[i]))/(float(len(trainX))))
    #
    # # Test Dataset Prediction Accuracy
    # test_prediction = predict(dictionary, testX, testY, class_vocab, class_prior)
    # test_acc = accuracy(testY, test_prediction)
    # print ("Test Set Accuracy:",test_acc)
    # # Confusion Matrices
    # testY_a = np.zeros(len(testY))
    # for i in range(len(testY)):
    #     testY_a[i] = testY[i]
    # confusion_mat = confusion_matrix(testY_a, test_prediction)
    # print (confusion_mat)
    # matrix_plot(confusion_mat)
    #
    # # Train Dataset Prediction Accuracy
    # train_prediction = predict(dictionary, trainX, trainY, class_vocab, class_prior)
    # train_acc = accuracy(trainY, train_prediction)
    # print ("Train Set Accuracy:",train_acc)

    # ************** UNCOMMENT TO RUN THE RANDOM PREDICTION MODEL **************
    # # Random prediction
    # rand_prediction = np.random.randint(1,6,(len(testX)))
    # rand_acc = accuracy(testY, rand_prediction)
    # print ("Random Prediction Accuracy:",rand_acc)

    # ************** UNCOMMENT TO RUN THE MAJORITY PREDICTION MODEL ************
    # # Majority Prediction
    # major = 1+np.argmax(class_occur)
    # major_prediction = [major]*(len(testY))
    # major_prediction = np.array(major_prediction)
    # major_acc = accuracy(testY, major_prediction)
    # print ("Majority Prediction Accuracy:",major_acc)

    # ************** UNCOMMENT TO RUN THE MODEL WITH STEMMING ******************
    # # Stemming the words
    # stem_dict,stem_class_occur,stem_class_vocab = stem_words(trainX,trainY)
    # # Log Likelihood Calculation
    # stem_dictionary = stem_dict
    # for word in stem_dictionary:
    #     for i in range(5):
    #         stem_dictionary[word][i] = math.log((float(stem_dictionary[word][i])+ 1)/(float(len(stem_dictionary) + stem_class_vocab[i])))
    # # Log Prior Calculation
    # stem_class_prior = [0.0,0.0,0.0,0.0,0.0]
    # for i in range(5):
    #     stem_class_prior[i] = math.log((float(stem_class_occur[i]))/(float(len(trainX))))
    # # Test Set Accuracy for stemmed word
    # stem_test_prediction = predict_for_stemmed(stem_dictionary, testX, testY, stem_class_vocab, stem_class_prior)
    # stem_test_acc = accuracy(testY, stem_test_prediction)
    # print ("Test Set Accuracy after Stemming:",stem_test_acc)

    # ****************** UNCOMMENT TO RUN MODEL WITH N-GRAMS *******************
    # Feature Engineering : bigrams
    start = time.time()
    b_dict,b_class_occur,b_class_vocab = bigram_words(trainX,trainY)
    #   Log Likelihood Calculation
    b_dictionary = b_dict
    for word in b_dictionary:
        for i in range(5):
            b_dictionary[word][i] = math.log((float(b_dictionary[word][i])+ 1)/(float(len(b_dictionary) + b_class_vocab[i])))
    #   Log Prior Calculation
    b_class_prior = [0.0,0.0,0.0,0.0,0.0]
    for i in range(5):
        b_class_prior[i] = math.log((float(b_class_occur[i]))/(float(len(trainX))))
    end = time.time()
    # print ("Training Time:", (end-start))
    #   Test Set Accuracy for stemmed word
    b_test_prediction = predict_for_bigram(b_dictionary, testX, testY, b_class_vocab, b_class_prior)
    # b_test_prediction = convert(output_filename)
    b_test_acc = accuracy(testY, b_test_prediction)
    # print (b_test_acc)
    f = open(output_filename, 'w')
    f.write(str(list(b_test_prediction.astype(int))))
    f.close()
    # print ("Test Set Accuracy after Trigrams:",b_test_acc)

    # ***************** UNCOMMENT TO RUN MODEL WITH LEMMATIZER *****************
    # # Feature Engineering : Lemmatizer
    # l_dict,l_class_occur,l_class_vocab = lemmatize_words(trainX,trainY)
    # #   Log Likelihood Calculation
    # l_dictionary = l_dict
    # for word in l_dictionary:
    #     for i in range(5):
    #         l_dictionary[word][i] = math.log((float(l_dictionary[word][i])+ 1)/(float(len(l_dictionary) + l_class_vocab[i])))
    # #   Log Prior Calculation
    # l_class_prior = [0.0,0.0,0.0,0.0,0.0]
    # for i in range(5):
    #     l_class_prior[i] = math.log((float(l_class_occur[i]))/(float(len(trainX))))
    # #   Test Set Accuracy for stemmed word
    # l_test_prediction = predict_for_lemmatizer(l_dictionary, testX, testY, l_class_vocab, l_class_prior)
    # l_test_acc = accuracy(testY, l_test_prediction)
    # print ("Test Set Accuracy after Lemmatizing:",l_test_acc)
if __name__ == "__main__":
    main()
