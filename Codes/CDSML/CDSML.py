# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 03:14:32 2020

@author: fahma
"""

# Importing required libraries

import sys
import copy
import numpy as np
import random
from manifold import manifold_learning
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


def normalise_sim(similarity_matrix):
       """ This function aims to normalize the similarity matrix
       using symmetric normalized Laplacian
       The input must be a square matrix
       """
       
       # change the type of matrix to a numpy matrix
       similarity_matrix = np.matrix(similarity_matrix)
       
       for round in range(200):
            # compute the summation of each line
            summ = np.sum(similarity_matrix, axis=1)
            a = np.matrix(summ)
            # construct a diagonal matrix with summation elements
            D = np.diag(a.A1) 
            # compute lagrange matrix
            D1 = np.linalg.pinv(np.sqrt(D)); 
            similarity_matrix = D1 * similarity_matrix * D1;
    
       return similarity_matrix

def modelEvaluation(real_matrix,predict_matrix,testPosition): 
       """ This function computes the evaluation criteria
       
       real_matrix: is a matrix with cell lines in rows, drugs in columns,
       and real labels in its elemnts
       
       predict_matrix: has the same size as the real matrix 
       with the predicted sensitivity probabilities
       
       testPosition: is a vecoto, containing the pairs of (i,j) indices of 
       cell line-drug pairs that were considered as the test samples 
       in cross validation
       """
       
       # gathering the test position values in real_matrix and predict_matrix into vectors
       
       real_labels=[]
       predicted_probability=[]
       for i in range(0,len(testPosition)):
           real_labels.append(real_matrix[testPosition[i,0], testPosition[i,1]])
           predicted_probability.append(predict_matrix[testPosition[i,0]][ testPosition[i,1]])
       real_labels = np.array(real_labels)
       predicted_probability=np.array(predicted_probability)
       predicted_probability=predicted_probability.reshape(-1,1)
       
       # computing AUPR criteria
       precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
       aupr_score = auc(recall, precision)
       
       # computing AUC criteria
       fpr, tpr, auc_thresholds = roc_curve(real_labels, predicted_probability)
       auc_score = auc(fpr, tpr)

       # computing best threshold
       all_F_measure=np.zeros(len(pr_thresholds))
       for k in range(0,len(pr_thresholds)):
           if (precision[k]+precision[k])>0:
              all_F_measure[k]=2*precision[k]*recall[k]/(precision[k]+recall[k])
           else:
              all_F_measure[k]=0
       max_index=all_F_measure.argmax()
       threshold =pr_thresholds[max_index]
       print(threshold)
       
       # binarize the predited probabilities       
       predicted_score=np.zeros(len(real_labels))
       predicted_score=np.where(predicted_probability > (threshold), 1, 0)

       # computing other criteria
       f=f1_score(real_labels,predicted_score)
       accuracy=accuracy_score(real_labels,predicted_score)
       precision=precision_score(real_labels,predicted_score)
       recall=recall_score(real_labels,predicted_score,predicted_probability)
       
       # gathering all computed criteria
       results=[auc_score, aupr_score, accuracy, f,precision,recall]
       return results


def runapp(label_file, simC_name, simD_name, percent, miu, landa, CV_num, repetition):
    
    """ This function runs the cross validtion
    
    label_file is the address and name of real labels file
    SimC_name and SimD_name are the address and name of similarity matrices
    of cell lines and drugs, respectively
    percent is the rank of latent matrix
    miu and landa are two model hyperparametrs.
    miu is the regularization coeeficient for latent matrices
    landa controls the similarity conservation while manifold learning
    CV_num is the number of folds in cross validation
    repetition is the number of repeting the cross validation
    """
    #-----------------------------------------------------------
    
    
    #reading label file
    R = np.loadtxt(label_file, dtype=float, delimiter=",") 
    
    # reading similarity matrices
    simD = np.loadtxt(simD_name, dtype=float, delimiter=",")
    simC = np.loadtxt(simC_name, dtype=float, delimiter=",")
    
    # constructing indices matrix
    seed = 0
    pos_number = 0
    all_position = []
    for i in range(0, len(R)):
            for j in range(0, len(R[0])):
                    pos_number = pos_number + 1
                    all_position.append([i, j])
    
    all_position = np.array(all_position)
    random.seed(seed)
    index = np.arange(0, pos_number)
    
    # shuffle the indices
    random.shuffle(index)
    
    # Compute the number of samples in each fold
    fold_num = (pos_number)// CV_num

    # initilaizing the evaluation criteria for all repetitions
    auc_cv = 0     
    aupr_cv = 0
    f_cv = 0
    acc_cv = 0    
    prc_cv = 0
    rec_cv = 0
    
    # repeating the cross valiation
    for rep in range(repetition):
        print('*********repetition:' + str(repetition) + "**********\n")
        
        # initializing the criteria value for every running of cross validation
        auc = 0     
        aupr = 0
        acc = 0
        f = 0 
        prc = 0
        rec = 0 
        
        # running the cross validation
        for CV in range(0, CV_num):
            print('*********round:' + str(CV) + "**********\n")
            
            # determining test positions
            test_index = index[(CV * fold_num):((CV + 1) * fold_num)]
            test_index.sort()
            testPosition = all_position[test_index]
            train_IC= copy.deepcopy(R)
            for i in range(0, len(testPosition)):
                train_IC[testPosition[i, 0], testPosition[i, 1]] = 0

            # initializng the latent matrices
            N = len(train_IC)
            M = len(train_IC[0])
            dim = min(N, M)
            K  = int(round (percent * dim))
            P = np.random.rand(N ,K)
            Q = np.random.rand(M, K)
            
            # calling the manifold learning on the sensitivity matrix
            predict_matrix1, A1,B1 = manifold_learning(train_IC, P, Q, K, simD, simC, landa, miu)
            
            # calling the manifold learning on the transpose of sensitivity matrix
            predict_matrix2, A2,B2 = manifold_learning(train_IC.T, B1, A1, K, simC, simD, landa, miu)
            
            # compute the final prediction matrix
            predict_matrix = 0.5 * (predict_matrix1 + predict_matrix2.T)
            
            # evaluate the predicted results
            results  = modelEvaluation(R, predict_matrix, testPosition)
            auc = auc + results[0]
            aupr = aupr + results[1]
            acc = acc + results[2]
            f = f + results[3]
            prc = prc + results[4]
            rec = rec + results[5]

        # averaging criteria over folds
        auc_cv = auc_cv + round(auc / CV_num, 4)
        aupr_cv = aupr_cv + round(aupr / CV_num, 4)
        f_cv = f_cv + round(f / CV_num, 4)
        acc_cv = acc_cv + round(acc / CV_num, 4)
        prc_cv = prc_cv + round(prc / CV_num, 4)
        rec_cv = rec_cv + round(rec / CV_num, 4)

    # averaging criteria over repetitions   
    auc_rep = round(auc_cv / repetition, 4)
    aupr_rep = round(aupr_cv / repetition, 4)
    f_rep = round(f_cv / repetition, 4)
    acc_rep = round(acc_cv / repetition, 4)
    prc_rep = round(prc_cv / repetition, 4)
    rec_rep = round(rec_cv / repetition, 4)
    
    print( auc_rep, ' ',aupr_rep, ' ',acc_rep, ' ', f_rep, ' ', prc_rep, ' ', rec_rep)
    
    
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
      elif key=='dim':
          percent=float(val)
      elif key=='miu':
          miu=float(val)
      elif key=='lambda':
          landa=float(val)
      elif key=='CV':
          CV_num=int(val)
      elif key=='repeat':
          repetition=int(val)
          
    # call the method
    runapp(label_file, simC_name, simD_name, percent, miu, landa, CV_num, repetition)
    
main()
