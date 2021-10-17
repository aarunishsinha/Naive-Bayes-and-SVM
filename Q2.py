import sys
import os
import csv
import time
import math
import numpy as np
import pandas as pd
import cvxopt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import svm

def convert(filename):
    f = open(filename, 'r')
    string = f.read().strip()[1:]
    string = string[:-1]
    preds = string.split(", ")
    preds = np.array(preds).astype(int)
    return preds

# Plotting the Confusion Matrix
def matrix_plot(confusion_mat, name):
    plt.imshow(confusion_mat)
    plt.title("Confusion Matrix for Test Set Prediction(Naive Bayes)")
    plt.colorbar()
    plt.set_cmap("Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(name,dpi=200)
    # plt.show()
# ******************* BINARY CLASSIFICATION ***************************
def initialise(path):
    data = np.array(pd.read_csv(path, header=None, dtype=float).values)
    X = []
    Y = []
    d = 1
    for i in range(len(data)):
        if (data[i][-1] == d) or (data[i][-1] == (d+1)):
            X.append(data[i][:-1])
            Y.append(data[i][-1])
    for i in range(len(Y)):
        if Y[i] == d:
            Y[i] = 1.0
        else:
            Y[i] = -1.0
    X = (np.array(X))/255
    Y = np.array(np.matrix(Y)).T
    X = np.asmatrix(X)
    Y = np.asmatrix(Y)
    # print (np.shape(Y))
    # print (np.shape(X))
    return X,Y

# SOLVING THE DUAL SVM PROBLEM USING LINEAR KERNEL
def lin_kernel(X, Y, penalty):
    XY = np.multiply(X,Y)
    P = cvxopt.matrix(np.dot(XY,XY.T))
    q = cvxopt.matrix(-1*np.ones((np.shape(X)[0],1)))
    A = cvxopt.matrix(Y.T)
    b = cvxopt.matrix(0.0)
    temp1 = -1*np.identity(np.shape(X)[0])
    temp2 = np.identity(np.shape(X)[0])
    G = cvxopt.matrix(np.vstack((temp1,temp2)))
    temp1 = np.zeros((np.shape(X)[0],1))
    temp2 = penalty*np.ones((np.shape(X)[0],1))
    h = cvxopt.matrix(np.vstack((temp1,temp2)))
    solution = cvxopt.solvers.qp(P,q,G,h,A,b)
    return solution
def lin_kernel_param(solution, trainX, trainY, tolerance):
    num_SV = 0
    m,n = np.shape(trainX)
    raveled = np.ravel(solution['x'])
    lagrangian_multipliers = np.arange(len(raveled)) [raveled>tolerance]
    w = np.zeros((1,n),dtype=float)
    for i in lagrangian_multipliers:
        for j in range(n):
            w[0,j] += (raveled[i]*trainX[i,j]*trainY[i,0])
        num_SV += 1
    b = 0
    if num_SV == 0:
        print("No support vector for the given tolerance")
    else:
        for sv_idx in lagrangian_multipliers:
            b += (trainY[sv_idx,0] - np.dot(trainX[sv_idx,:],w.T)[0])
        b = b/(float(len(lagrangian_multipliers)))
        print("value of b = ", b)
    return w,b,num_SV
def lin_kernel_predict(w,b,testX):
    predicted = np.zeros((len(testX),1),dtype=int)
    value = np.dot(testX,w.T)+b
    predicted = 2*np.multiply((value>0),np.ones((len(testX),1))) - 1
    return predicted
def linear_kernel_model_BC():
    trainX,trainY = initialise("fashion_mnist/train.csv")
    validationX,validationY = initialise("fashion_mnist/val.csv")
    testX,testY = initialise("fashion_mnist/test.csv")
    tolerance = 1e-4
    penalty = 1
    lin_ker_sol = lin_kernel(trainX,trainY,penalty)
    w,b,num_SV = lin_kernel_param(lin_ker_sol, trainX, trainY, tolerance)
    print ("Number of Suppurt Vectors = ",num_SV)
    val_predicted = lin_kernel_predict(w,b,validationX)
    test_predicted = lin_kernel_predict(w,b,testX)
    val_con = confusion_matrix(validationY, val_predicted)
    val_accuracy = np.trace(val_con)/len(validationY)
    print (val_accuracy)
    test_con = confusion_matrix(testY, test_predicted)
    test_accuracy = np.trace(test_con)/len(testY)
    print (test_accuracy)

# SOLVING DUAL SVM PROBLEM USING GAUSSIAN CLASSFICATION
def gauss_kernel(X, Y, gamma, penalty):
    kernel = np.matrix(np.zeros((np.shape(X)[0],np.shape(X)[0]),dtype=float))
    X_XT = np.dot(X, X.T)
    for i in range(np.shape(X)[0]):
        for j in range(np.shape(X)[0]):
            kernel[i,j] = float(X_XT[i,i] + X_XT[j,j] - 2*X_XT[i,j])
    kernel = np.exp(-1*gamma*kernel)
    P = cvxopt.matrix(np.multiply(kernel,np.dot(Y,Y.T)))
    q = cvxopt.matrix(-1*np.ones((np.shape(X)[0],1)))
    A = cvxopt.matrix(Y.T)
    b = cvxopt.matrix(0.0)
    temp1 = -1*np.identity(np.shape(X)[0])
    temp2 = np.identity(np.shape(X)[0])
    G = cvxopt.matrix(np.vstack((temp1,temp2)))
    temp1 = np.zeros((np.shape(X)[0],1))
    temp2 = penalty*np.ones((np.shape(X)[0],1))
    h = cvxopt.matrix(np.vstack((temp1,temp2)))
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P,q,G,h,A,b,show_progress=False)
    return solution
def gauss_kernel_predict(solution, trainX, trainY, testX,tolerance,gamma):
    # m,n = np.shape(trainX)
    # raveled = np.ravel(solution['x'])
    raveled = solution
    num_SV = 0
    train = np.sum(np.multiply(trainX,trainX),axis=1)
    # print ("Shape of trainX:",np.shape(trainX))
    test = np.sum(np.multiply(testX,testX),axis=1)
    # print ("Shape of testX:",np.shape(testX))
    train_test = np.dot(trainX,testX.T)
    # print ("Shape of trainY:",np.shape(trainY))
    alpha_x_label = np.matrix(np.zeros((len(raveled),1),dtype=float))
    for i in range(len(raveled)):
        if raveled[i]>tolerance:
            alpha_x_label[i,0] = trainY[i,0]*raveled[i]
            num_SV += 1
    lagrangian_multipliers = np.arange(len(raveled)) [raveled>tolerance]
    prediction = np.zeros((np.shape(testX)[0],1),dtype=int)
    if len(lagrangian_multipliers)==0:
        print("No support vectors for the current tolerance")
    else:
        b = 0
        for sv_idx in lagrangian_multipliers:
            b += (trainY[sv_idx,0] - np.sum(np.multiply(alpha_x_label,np.exp(-1*gamma*np.sum(np.multiply(trainX-trainX[sv_idx,:],trainX-trainX[sv_idx,:]),axis=1)))))
        b = b/(float(len(lagrangian_multipliers)))
        # print ("Value of b =",b)
        for i in range(np.shape(testX)[0]):
            prediction[i] = np.sign(np.sum(np.multiply(alpha_x_label,np.exp(-1*gamma*(train - 2*train_test[:,i] + test[i,0])))) + b)
    # print ("Number of Support Vectors =",num_SV)
    return prediction
def gaussian_kernel_model_BC(train_filename,test_filename,output_filename):
    trainX,trainY = initialise(train_filename)
    validationX,validationY = initialise("fashion_mnist/val.csv")
    testX,testY = initialise(test_filename)
    tolerance = 1e-4
    penalty = 1
    gamma = 0.05
    gauss_ker_sol = gauss_kernel(trainX, trainY, gamma, penalty)
    val_predicted = gauss_kernel_predict(gauss_ker_sol,trainX,trainY,validationX,tolerance,gamma)
    val_con = confusion_matrix(validationY, val_predicted)
    val_accuracy = np.trace(val_con)/len(validationY)
    print ("Validation:",val_accuracy)
    # print (val_con)
    test_predicted = gauss_kernel_predict(gauss_ker_sol,trainX,trainY,testX,tolerance,gamma)
    test_con = confusion_matrix(testY, test_predicted)
    test_accuracy = np.trace(test_con)/len(testY)
    print ("Test:",test_accuracy)

# *********************** MULTICLASS CLASSIFICATION *************************
# kC2 classifiers - Gaussian Kernel
def store_data(path):
    data = np.array(pd.read_csv(path, header=None, dtype=float).values)
    data_label = []
    for i in range(10):
        X = []
        Y = []
        for j in range(len(data)):
            if data[j][-1] == i:
                X.append(data[j][:-1])
                Y.append(data[j][-1])
        X = (np.array(X))/255
        Y = np.array(np.matrix(Y)).T
        X = np.asmatrix(X)
        Y = np.asmatrix(Y)
        data_label.append([X,Y])
    return data_label
def convert_labels(data_label, i, j):
    Y = np.vstack((data_label[i][1],data_label[j][1]))
    for k in range(np.shape(Y)[0]):
        if Y[k,0] == i:
            Y[k,0] = 1.0
        else:
            Y[k,0] = -1.0
    return Y
def read_test_data(path):
    data = np.array(pd.read_csv(path, header=None, dtype=float).values)
    X = np.array(data[:,0:784])/255
    X = np.asmatrix(X)
    Y = np.array(data[:,784:785])
    Y = np.asmatrix(Y)
    return X,Y

def gaussian_kernel_kC2(train_filename,test_filename,output_filename):
    gamma = 0.05
    penalty = 1
    tolerance = 1e-6
    f = open(output_filename, 'w')
    all_data_label = store_data(train_filename)
    svm_dict = {}
    # start = time.time()
    for i in range(10):
        for j in range(i):
            key = str(i)+str(j)
            svm_dict[key] = []
            trainX = np.vstack((all_data_label[i][0],all_data_label[j][0]))
            # print ("Shape of trainX:",np.shape(trainX))
            trainY = convert_labels(all_data_label,i,j)
            # trainY = np.ravel(trainY).tolist()
            # trainY = np.array(trainY)
            # print ("Shape of trainY:",np.shape(trainY))
            solution = gauss_kernel(trainX, trainY, gamma, penalty)
            svm_dict[key] = np.ravel(solution['x'])
    # end = time.time()
    # print ('Training Time:', (end-start))

    # print ('Running On Validation')
    # testX,testY = read_test_data("fashion_mnist/val.csv")
    # # testY = np.ravel(testY).tolist()
    # # testY = np.array(testY)
    # prediction_dict = {}
    # for i in range(len(testX)):
    #     prediction_dict[i] = [0,0,0,0,0,0,0,0,0,0]
    # prediction = np.matrix(np.zeros((len(testX),1),dtype=int))
    #
    # for i in range(10):
    #     for j in range(i):
    #         key = str(i)+str(j)
    #         solution = svm_dict[key]
    #         trainX = np.vstack((all_data_label[i][0],all_data_label[j][0]))
    #         # print ("Shape of trainX:",np.shape(trainX))
    #         trainY = convert_labels(all_data_label,i,j)
    #         # trainY = np.ravel(trainY).tolist()
    #         # trainY = np.array(trainY)
    #         # print ("Shape of trainY:",np.shape(trainY))
    #         svm_pred = gauss_kernel_predict(solution,trainX,trainY,testX,tolerance,gamma)
    #         print ("Lagrange :"+key+"done")
    #         for k in range(len(svm_pred)):
    #             if svm_pred[k,0] == 1:
    #                 prediction_dict[k][i] += 1
    #             else:
    #                 prediction_dict[k][j] += 1
    # for i in range(len(testX)):
    #     prediction[i] = np.argmax(prediction_dict[i])
    # conf = confusion_matrix(testY, prediction)
    # print (conf)
    # matrix_plot(conf,"Q2bi_val.png")

    # print ('Running on Test')
    testX,testY = read_test_data(test_filename)
    # testY = np.ravel(testY).tolist()
    # testY = np.array(testY)
    prediction_dict = {}
    for i in range(len(testX)):
        prediction_dict[i] = [0]*10
    prediction = np.matrix(np.zeros((len(testX),1),dtype=int))

    for i in range(10):
        for j in range(i):
            key = str(i)+str(j)
            solution = svm_dict[key]
            trainX = np.vstack((all_data_label[i][0],all_data_label[j][0]))
            # print ("Shape of trainX:",np.shape(trainX))
            trainY = convert_labels(all_data_label,i,j)
            # trainY = np.ravel(trainY).tolist()
            # trainY = np.array(trainY)
            # print ("Shape of trainY:",np.shape(trainY))
            svm_pred = gauss_kernel_predict(solution,trainX,trainY,testX,tolerance,gamma)
            for k in range(len(svm_pred)):
                if svm_pred[k,0] == 1:
                    prediction_dict[k][i] += 1
                else:
                    prediction_dict[k][j] += 1
    for i in range(len(testX)):
        prediction[i] = np.argmax(prediction_dict[i])
    f.write(str(np.ravel(np.array(prediction.T).astype(int)).tolist()))
    f.close()
    pred_read = convert(output_filename)

    conf = confusion_matrix(testY, pred_read)
    print (conf)
    # matrix_plot(conf,"Q2bi_test.png")

# Scikit SVM Libraries
def svm_classifier():
    data = np.array(pd.read_csv("fashion_mnist/train.csv", header=None, dtype=float).values)
    trainX = data[:,0:784]
    trainX = trainX/255
    trainX = np.asmatrix(trainX)
    trainY = data[:,784:785]
    trainY = np.ravel(np.asmatrix(trainY)).tolist()
    # n_estimators = 10
    start = time.time()
    gaussian_classifier = svm.SVC(C=1.0,kernel='rbf',gamma=0.05,tol=1e-4)
    # gaussian_classifier = BaggingClassifier(base_estimator = svm.SVC(C=1.0,kernel='rbf',gamma=0.05,tol=1e-5), max_samples=1.0 / n_estimators, n_estimators=n_estimators,bootstrap = False, n_jobs=-1)
    gaussian_classifier.fit(trainX,trainY)
    end = time.time()
    # print ("Training Time:", (end-start))

    # data = np.array(pd.read_csv("fashion_mnist/val.csv", header=None, dtype=float).values)
    # validationX = data[:,0:784]
    # validationX = validationX/255
    # validationX = np.asmatrix(validationX)
    # validationY = data[:,784:785]
    # validationY = np.asmatrix(validationY)
    # val_pred = gaussian_classifier.predict(validationX)
    # val_con = confusion_matrix(validationY,val_pred)
    # # matrix_plot(val_con, "svm_rbf_val.png")
    # print ("Confusion Matrix for validation set:",val_con)
    # val_accuracy = np.trace(val_con)/len(validationY)
    # print ("Validation Accuracy:",val_accuracy)

    data = np.array(pd.read_csv("fashion_mnist/test.csv", header=None, dtype=float).values)
    testX = data[:,0:784]
    testX = testX/255
    testX = np.asmatrix(testX)
    testY = data[:,784:785]
    testY = np.asmatrix(testY)
    test_pred = gaussian_classifier.predict(testX)
    test_con = confusion_matrix(testY,test_pred)
    # matrix_plot(test_con, "svm_rbf_test.png")
    # print ("Confusion Matrix for test set:",test_con)
    test_accuracy = np.trace(test_con)/len(testY)
    # print ("Test Accuracy:",test_accuracy)

# K-fold CROSS VALIDATION
def cross_validation():
    data = np.array(pd.read_csv("fashion_mnist/train.csv", header=None, dtype=float).values)
    trainX = data[:,0:784]
    trainX = trainX/255
    trainX = np.asmatrix(trainX)
    trainY = data[:,784:785]
    trainY = np.asmatrix(trainY)
    data = np.array(pd.read_csv("fashion_mnist/test.csv", header=None, dtype=float).values)
    testX = data[:,0:784]
    testX = testX/255
    testX = np.asmatrix(testX)
    testY = data[:,784:785]
    testY = np.asmatrix(testY)
    Cs = [1e-5,1e-3,1,5,10]
    val_accs = []
    test_accs = []
    for c in Cs:
        # print ("Running for c =",c)
        gaussian_classifier = svm.SVC(C=c,kernel='rbf',gamma=0.05, tol=1e-4)
        # n_estimators = 10
        # gaussian_classifier = BaggingClassifier(base_estimator = svm.SVC(C=1.0,kernel='rbf',gamma=0.05,tol=1e-4), max_samples=1.0 / n_estimators, n_estimators=n_estimators,bootstrap = False, n_jobs=-1)
        vals = []
        tests = []
        kf = KFold(n_splits=5)
        kf.get_n_splits(trainX)
        # for i in range(5):
        for train_index, val_index in kf.split(trainX):
            # print ("Creating Datasets...")
            X_train,X_val = trainX[train_index], trainX[val_index]
            Y_train,Y_val = trainY[train_index], trainY[val_index]
            Y_train = np.ravel(Y_train).tolist()
            # trainX,valX,trainY,valY = train_test_split(trainX,trainY, test_size=0.2)
            # print ("Fitting...")
            gaussian_classifier.fit(X_train,Y_train)
            # print ("Predicting for Validation...")
            val_pred = gaussian_classifier.predict(X_val)
            val_con = confusion_matrix(Y_val,val_pred)
            val_accuracy = np.trace(val_con)/len(Y_val)
            vals.append(100*val_accuracy)
            # print ("Validation accuracy:",val_accuracy)
            # print ("Predicting for Test...")
            test_pred = gaussian_classifier.predict(testX)
            test_con = confusion_matrix(testY,test_pred)
            test_accuracy = np.trace(test_con)/len(testY)
            tests.append(100*test_accuracy)
        #     print ("Test accuracy:",test_accuracy)
        # print ('Storing max accuracies...')
        val_accs.append(max(vals))
        test_accs.append(max(tests))
    # print ("Plotting...")
    for i in range(len(Cs)):
        Cs[i] = math.log(Cs[i],10)
    plt.figure()
    plt.xlabel("logC")
    plt.ylabel("Accuracy")
    plt.plot(Cs,val_accs, label="Validation Set Accuracy")
    plt.plot(Cs,test_accs, label="Test Set Accuracy")
    plt.legend()
    plt.savefig("cross_val.png",dpi=200)
    # plt.show()
def main():
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    output_filename = sys.argv[3]
    # linear_kernel_model_BC()
    # start = time.time()
    # gaussian_kernel_model_BC(train_filename,test_filename,output_filename)
    gaussian_kernel_kC2(train_filename,test_filename,output_filename)
    # end = time.time()
    # print ('Training Time:', (end-start))
    # svm_classifier()
    # cross_validation()
if __name__ == "__main__":
    main()
