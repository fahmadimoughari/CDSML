# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 04:47:13 2020

@author: fahma
"""


# Importing required libraries

import sys
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def model_evaluation(real_labels,predicted):
    """ This function computes the evaluation criteria
       
       real_labels: is a vector of real labels
       
       predicted: has the same size as the real_labels 
       with the predicted probabilities
       """
    predicted_probability=[]
    for i in range(len(predicted)):
        # choose the probability of belonging to the sensitivity class
        predicted_probability.append(predicted[i][1])
    predicted_probability=np.array(predicted_probability)
    
    # compute AUC
    fpr, tpr, auc_thresholds = roc_curve(real_labels, predicted_probability)
    auc_score = auc(fpr, tpr)
    return auc_score
    

def prepare_data(interaction, drug_fea,cell_fea,split):
    """ Split the whole data into test/train split """
    
    # define necessary variables
    train = []
    label = []
    test=[]
    label_test=[]
    tmp_fea=[]
    num_samples=int(interaction.shape[0]*interaction.shape[1]*split)
    count=0
    
    
    for i in range(0, interaction.shape[0]):
        for j in range(0, interaction.shape[1]):
            
            # concatenating cell line and drug similarities for each pair
            tmp_fea=[]
            for k in range(len(cell_fea[i])):
                tmp_fea.append(cell_fea[i][k])
            for k in range(len(drug_fea[j])):
                tmp_fea.append(drug_fea[j][k])
                
            if count<num_samples:
                label.append(interaction[i,j])
                train.append(tmp_fea)
            else:
                test.append(tmp_fea)
                label_test.append(interaction[i,j])
            count+=1

    y_train=np.array(label)
    y_test=np.array(label_test)
    X_train=np.array(train)
    X_test=np.array(test)
    return X_train, y_train, X_test, y_test

def runapp(label_file, simC_name, simD_name,cName):
    
    """ This function runs the model tuning
    
    label_file is the address and name of real labels
    SimC_name and SimD_name are the address and name of similarity matrices
    of cell lines and drugs, respectively
    cName is the name of machine learning model
    """
    #-----------------------------------------------------------
    
    
    #reading label file
    R = np.loadtxt(label_file, dtype=float, delimiter=",")      
        
    # reading similarity matrices
    simD = np.loadtxt(simD_name, dtype=float, delimiter=",")
    simC = np.loadtxt(simC_name, dtype=float, delimiter=",")
    
    # split data into %70 for training and %30 for testing
    X_train, y_train, X_test, y_test = prepare_data(R,simD,simC, 0.7)

    best_eval=0
    best_param1=0
    best_param2=0
    
    # tuning Guassian Naiive Bayes model
    if cName=='GNB':
        
        # Considered values for variance smoothing
        param1=[1e-12,1e-9,1e-6,1e-3,1e-1]

        for p1 in param1:
                # Consruct the model
                clf=GaussianNB(var_smoothing=p1)
                
                # Train the model
                clf.fit(X_train, y_train)
                
                # Predict the model results on test data
                pred=clf.predict_proba(X_test)
                
                # Compute AUC of model
                auc=model_evaluation(y_test,pred)
                
                # Update the best hyper-parameter
                if auc>best_eval:
                    best_eval=auc
                    best_param1=p1
                    
        print('GNB','best var',best_param1,'best_eva', best_eval)

    # tuning Logistic Regression model   
    if cName=='LR':
        
        # Considered values for regularization scale
        param1=[0.001, 0.1,1, 0.01, 10, 100]

        # Considered values for stop tolerance
        param2=[1e-6,1e-4,1e-2]

        for p1 in param1:
            for p2 in param2:
                
                # Consruct the model
                clf=LogisticRegression(max_iter=1000, penalty='l2', tol=p2, C=p1)
                
                # Train the model
                clf.fit(X_train, y_train)
                
                # Predict the model results on test data
                pred=clf.predict_proba(X_test)
                pred=np.array(pred)
                preds=pred[:,1]

                # Compute AUC of model
                auc=model_evaluation(y_test,preds)
                
                # Update the best hyper-parameter
                if auc>best_eval:
                    best_eval=auc
                    best_param1=p1
                    best_param2=p2
                    
        print('LR','best C',best_param1,'best_t',best_param2,'best_eva', best_eval)
    
    # tuning Random Forest model
    if cName=='RF':
        
        # Considered values for number of trees
        param1=[10,50,100,500,1000]

        # Considered values for criterion
        param2=["gini", "entropy"]

        for p1 in param1:
            for p2 in param2:
                
                # Consruct the model
                clf=RandomForestClassifier(n_estimators=p1, criterion=p2)
                
                # Train the model
                clf.fit(X_train, y_train)
                
                # Predict the model results on test data
                pred=clf.predict_proba(X_test)
                
                # Compute AUC of model
                auc=model_evaluation(y_test,pred)
                
                # Update the best hyper-parameter                
                if auc>best_eval:
                    best_eval=auc
                    best_param1=p1
                    best_param2=p2
                    
        print('RF','best num_estimator',best_param1,'best_criterion',best_param2,'best_eva', best_eval)

    # tuning Support vector machine model
    if cName=='SVM':
        param1=["linear", "poly", "rbf", "sigmoid", "precomputed"]
        param2=[0.01,0.1,1,10,100]

        for p1 in param1:
                
                # Consruct the model
                clf=LinearSVC(C=p1, tol=1e-5)
                
                # Train the model
                clf.fit(X_train, y_train)
                
                # Predict the model results on test data
                pred=clf.decision_function(X_test)
                
                # Compute AUC of model
                auc=model_evaluation(y_test,pred)
                
                # Update the best hyper-parameter  
                if auc>best_eval:
                    best_eval=auc
                    best_param=p1
                    
        print('SVM','bestC',best_param,'best_eva', best_eval)

    # tuning k-nearest neighbors model
    if cName=='KNN':
        param1=[15,17,19,21,23,25]
        for p1 in param1:
                
                # Consruct the model
                clf=KNeighborsClassifier(n_neighbors=p1)
                
                # Train the model
                clf.fit(X_train, y_train)
                
                # Predict the model results on test data
                pred=clf.predict(X_test)
                
                # Compute AUC of model
                auc=model_evaluation(y_test,pred)
                
                # Update the best hyper-parameter  
                if auc>best_eval:
                    best_eval=auc
                    best_param1=p1
                    
        print('KNN','best n_neighbors',best_param1,'best_eva', best_eval)

    # tuning Adaptive boosting model
    if cName=='Ada':
        param1=[10,50,100,500,1000]
        param2=[1,1.25,1.5,1.75,2]
        for p in param1:
            for lr in param2:
                
                # Consruct the model
                clf = AdaBoostClassifier(n_estimators=p1,learning_rate=p2)
                
                # Train the model
                clf.fit(X_train, y_train)
                
                # Predict the model results on test data
                pred=clf.predict(X_test)
                
                # Compute AUC of model
                auc=model_evaluation(y_test,pred)
                
                # Update the best hyper-parameter  
                if auc>best_eval:
                    best_eval=auc
                    best_param1=p1
                    best_param2=p2
                    
        print('Adaboost','best n_estimators',best_param,'learning_rate',best_param2,'best_eva', best_eval)

    # tuning Multi layer perceptron model    
    if cName=='MLP':
        param1= [(50,50,50), (50,100,50), (100,)]
        best_param3=0
        best_param4=0
        best_param5=0
        param2=['tanh', 'relu']
        param3=['sgd', 'adam']
        param4=[0.0001, 0.05]
        param5=['constant','adaptive']
        for p1 in param1:
            for p2 in param2:
                for p3 in param3:
                    for p4 in param4:
                        for p5 in param5:
                            
                            # Consruct the model
                            clf=MLPClassifier(max_iter=100,hidden_layer_sizes=p1,activation=p2,solver=p3,alpha=p4,learning_rate=p5)
                
                            # Train the model
                            clf.fit(X_train, y_train)
                
                            # Predict the model results on test data
                            pred=clf.predict(X_test)
                
                            # Compute AUC of model
                            auc=model_evaluation(y_test,pred)
                
                            # Update the best hyper-parameter  
                            if auc>best_eval:
                                best_eval=auc
                                best_param1=p1
                                best_param2=p2
                                best_param3=p3
                                best_param4=p4
                                best_param5=p5
                                
        print('MLP','best hidden_layer_sizes',best_param1,'best_activation',best_param2,'solver',best_param3,'alpha',best_param4,'learning_rate',best_param5,'best_eva', best_eval)
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
          
    # call the method
    methods=['GNB','LR','RF','SVM','KNN','Ada','MLP']
    for method in methods:
        runapp(label_file, simC_name, simD_name, method)

main()
            
