# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 05:46:17 2020

@author: fahma
"""

# Importing required libraries
import sys
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
import os.path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from keras.utils import np_utils


def prepare_data(interaction, drug_fea,cell_fea):
    """ Prepare the data by concatenating cell line similarity and drug
    similarity for each cell line-drug pair"""
    
    # define necessary variables
    feature = []
    label = []

    for i in range(0, interaction.shape[0]):
        for j in range(0, interaction.shape[1]):
            tmp_fea=[]
            # concatenating cell line and drug similarities for each pair
            for k in range(len(cell_fea[i])):
                tmp_fea.append(cell_fea[i][k])
            for k in range(len(drug_fea[j])):
                tmp_fea.append(drug_fea[j][k])
            label.append(interaction[i,j])
            feature.append(tmp_fea)
            
    label=np.array(label)
    feature=np.array(feature)
    return feature, label
    
def modelEvaluation(real_labels,predicted_probability):
    
       """ This function computes the AUC and AUPR criteria
       
       real_labels: is a vector of real labels 
       
       predicted_probability: has the same size as the real_labels 
       with the predicted probabilities
       """
      
       real_labels = np.array(real_labels)
       predicted_probability=np.array(predicted_probability)
       predicted_probability=predicted_probability.reshape(-1,1)
       
       # computing AUC and AUPR criteria
       fpr, tpr, auc_thresholds = roc_curve(real_labels, predicted_probability)
       auc_score = auc(fpr, tpr)
       precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
       aupr_score = auc(recall, precision)
       
       results=[auc_score, aupr_score]
       return results
       
def model_eval(real_labels,predicted_score):
    
       """ This function computes the classification criteria
       
       real_labels: is a vector of real labels 
       
       predicted_score: has the same size as the real_labels 
       with the predicted labels
       """
       
       f=f1_score(real_labels,predicted_score)
       accuracy=accuracy_score(real_labels,predicted_score)
       precision=precision_score(real_labels,predicted_score)
       recall=recall_score(real_labels,predicted_score)
       results=[accuracy, f, precision, recall]
       return results

def runapp(cName):
    
    """ This function runs the cross validtion on the the machine learning 
    models using their best tuned hyper parameters
    
    cName is the name of machine learning model
    """
    
    # Constructing Gaussian Naiive Bayes model
    if cName=='GNB':
        p=0.1
        model=GaussianNB(var_smoothing=p)
        
    # Constructing Logistic Regression model
    if cName=='LR':
        t=1e-06 
        p=1
        model=LogisticRegression(max_iter=1000, penalty='l2', tol=t, C=p)
            
    # Constructing Random Forest model
    if cName=='RF':
        c='entropy'
        p=100
        model=RandomForestClassifier(n_estimators=p, criterion=c)
            
    # Constructing Adaptive Boosting model
    if cName=='Ada':
        model=AdaBoostClassifier(n_estimators=50,learning_rate=1.25)
            
    # Constructing k-nearest neighbor model
    if cName=='KNN':
        model=KNeighborsClassifier(n_neighbors=19)
            
    # Constructing Multi layer Perceptron model
    if cName=='MLP':
        model=MLPClassifier(max_iter=100,hidden_layer_sizes=(50,50,50),activation='relu',solver='adam',alpha=0.05,learning_rate='adaptive')
            
    # Constructing Support Vector Machinne model
    if cName=='SVM':
        model=LinearSVC(C=0.1, tol=1e-5)
        
    return model

def main():
    # get the options from user
    for arg in sys.argv[1:]:
      (key,val) = arg.rstrip().split('=')
      if key == 'label_file':
          label_file=val
      elif key=='simC_dirc':
          simC_name=val
      elif key=='simD_dirc':
          simD_name=val
    
    # Read label file and similarity files
    R = np.loadtxt(label_file, dtype=float, delimiter=",")      
    simD = np.loadtxt(simC_name, dtype=float, delimiter=",")
    simC = np.loadtxt(simD_name, dtype=float, delimiter=",")
    
    # Concatenate the cell line and drug similarities for each pair
    X,y = prepare_data(R,simD,simC)
    
    methods=['GNB','LR','SVM','RF','MLP','Ada','KNN']
    
    CV_num=5
    seed=0
    
    # compute the number of samples in each fold
    fold_num=int(X.shape[0]/CV_num)
    random.seed(seed)
    index = np.arange(0, X.shape[0])
    
    # Shuffle the samples
    random.shuffle(index)
    
    for method in methods:
        
        #Construct the model
        model=runapp(method)
        
        # Initialize the evaluation criteria for all folds
        acc=0
        pre=0
        rec=0
        f=0
        aucf=0
        aupr=0
        
        for CV in range(0, CV_num):
            
                # seleting test positions
                test_index = index[(CV * fold_num):((CV + 1) * fold_num)]
                X_train=[]
                X_test=[]
                y_train=[]
                y_test=[]
    
                # Set the test and train data
                for i in range(X.shape[0]):
                    if i in test_index:
                        X_test.append(X[i])
                        y_test.append(y[i])
                    else:
                        X_train.append(X[i])
                        y_train.append(y[i])
               
                # Training the model on train data
                model.fit(X_train,y_train)
                
                # compute the predicted probabillities on test data
                if method=='SVM':
                    y_tr=model.decision_function(X_train)
                    y_p=model.decision_function(X_test)
                else:
                    y_tra=model.predict_proba(X_train)
                    y_tr=[]
                    for i in range(len(y_tra)):
                        y_tr.append(y_tra[i][1])
                    y_pred=model.predict_proba(X_test)
                    y_p=[]
                    for i in range(len(y_pred)):
                        y_p.append(y_pred[i][1])
                
                # compute the predicted labels on test data
                y_score=model.predict(X_test)
                
                # compute AUC nad AUPR criteria
                res=model_eval(y_test,y_score)
                
                # compute other criteria
                results=modelEvaluation(y_test,y_p,y_train, y_tr)
                
                
                aucf=aucf+results[0]
                aupr=aupr+results[1]
                acc=acc+res[0]
                f=f+res[1]
                pre=pre+res[2]
                rec=rec+res[3]
    
        print(method,round(aucf/CV_num,4),round(aupr/CV_num,4),round(acc/CV_num,4),round(f/CV_num,4),round(pre/CV_num,4),round(rec/CV_num,4))


main()        
