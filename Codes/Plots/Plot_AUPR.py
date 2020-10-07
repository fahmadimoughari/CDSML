# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 01:09:15 2020

@author: fahma
"""

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import numpy as np
from sklearn.metrics import auc
real_matrix=np.loadtxt('..\..\Data\B_Sensitivity.csv',dtype=float, delimiter=",")
reals=[]
CV_num=5
def construct_vec(real_matrix,testPosition):
    real_labels=[]
    for i in range(0,len(testPosition)):
        real_labels.append(real_matrix[int(testPosition[i,0]),int(testPosition[i,1])])
           

    real_labels = np.array(real_labels)
    return real_labels
for CV in range(0, CV_num):
                    ide=np.loadtxt('id_test'+str(CV)+'.csv', dtype=float, delimiter=",")
                    reals.append(construct_vec(real_matrix,ide))
def retrun_CV(k):                   
        label_i=reals[k]
        return  label_i
def prepare_dataa(cv):
    y_train=[]
    y_test=[]
    
    for k in range(cv-1):
        label=retrun_CV(k)
        for i in range(len(label)):
            y_train.append(label[i])
    label=retrun_CV(cv)
    for i in range(len(label)):
            y_test.append(label[i])
    for k in range(cv+1,5):
        label=retrun_CV(k)
        for i in range(len(label)):
            y_train.append(label[i])
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    return y_train, y_test
y_train, y_test = prepare_dataa(0)
CV=0
file_name='GDSC_fold_'+str(CV)+'_'
file_name_cnv=file_name+'rna1.csv'
cnv=[]
y_probas = np.loadtxt(file_name_cnv, dtype=float, delimiter=",")
ide=np.loadtxt('id_test'+str(CV)+'.csv', dtype=float, delimiter=",")
cnv.append(construct_vec(y_probas,ide))
predicted_probability=np.array(cnv)
predicted_probability=np.reshape(predicted_probability,(predicted_probability.shape[1]))
real_labels=np.array(y_test)

print(predicted_probability.shape,real_labels.shape)
precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
aupr_score = auc(recall, precision)

plt.plot(recall,precision,label="CDSML (AUPR="+str(round(aupr_score,3))+")", color = 'navy')

methods=['GNB','LR','RF','MLP','Ada','KNN','SVM']
GNB=[]
LR=[]
RF=[]
MLP=[]
Ada=[]
KNN=[]
SVM=[]
file_name_method=methods[0]+'_'+str(CV)+'.csv'
a1=np.loadtxt(file_name_method, dtype=float, delimiter=",")
file_name_method=methods[1]+'_'+str(CV)+'.csv'
a2=np.loadtxt(file_name_method, dtype=float, delimiter=",")
file_name_method=methods[2]+'_'+str(CV)+'.csv'
a3=np.loadtxt(file_name_method, dtype=float, delimiter=",")
file_name_method=methods[3]+'_'+str(CV)+'.csv'
a4=np.loadtxt(file_name_method, dtype=float, delimiter=",")
file_name_method=methods[4]+'_'+str(CV)+'.csv'
a5=np.loadtxt(file_name_method, dtype=float, delimiter=",")
file_name_method=methods[5]+'_'+str(CV)+'.csv'
a6=np.loadtxt(file_name_method, dtype=float, delimiter=",")
file_name_method=methods[6]+'__'+str(CV)+'.csv'
a7=np.loadtxt(file_name_method, dtype=float, delimiter=",")
GNB.append(a1[:,1])
LR.append(a2[:,1])
RF.append(a3[:,1])
MLP.append(a4[:,1])
Ada.append(a5[:,1])
KNN.append(a6[:,1])
SVM.append(a7)
predicted_probability=np.array(GNB)
predicted_probability=np.reshape(predicted_probability,(predicted_probability.shape[1]))

precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
aupr_score = auc(recall, precision)
plt.plot(recall,precision,label="GNB (AUPR="+str(round(aupr_score,3))+")", color = 'orange')

predicted_probability=np.array(LR)
predicted_probability=np.reshape(predicted_probability,(predicted_probability.shape[1]))

precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
aupr_score = auc(recall, precision)
plt.plot(recall,precision,label="LR (AUPR="+str(round(aupr_score,3))+")", color = 'dodgerblue')

predicted_probability=np.array(RF)
predicted_probability=np.reshape(predicted_probability,(predicted_probability.shape[1]))

precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
aupr_score = auc(recall, precision)
plt.plot(recall,precision,label="RF (AUPR="+str(round(aupr_score,3))+")", color = 'hotpink')


predicted_probability=np.array(MLP)
predicted_probability=np.reshape(predicted_probability,(predicted_probability.shape[1]))

precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
aupr_score = auc(recall, precision)
plt.plot(recall,precision,label="MLP (AUPR="+str(round(aupr_score,3))+")", color = 'brown')

predicted_probability=np.array(Ada)
predicted_probability=np.reshape(predicted_probability,(predicted_probability.shape[1]))

precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
aupr_score = auc(recall, precision)
plt.plot(recall,precision,label="ADA (AUPR="+str(round(aupr_score,3))+")", color = 'lime')

predicted_probability=np.array(KNN)
predicted_probability=np.reshape(predicted_probability,(predicted_probability.shape[1]))

precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
aupr_score = auc(recall, precision)
plt.plot(recall,precision,label="KNN (AUPR="+str(round(aupr_score,3))+")", color = 'red')

predicted_probability=np.array(SVM)
print(predicted_probability.shape,real_labels.shape )
predicted_probability=np.reshape(predicted_probability,(predicted_probability.shape[1]))

precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)

aupr_score = auc(recall, precision)
plt.plot(recall,precision,label="SVM (AUPR="+str(round(aupr_score,3))+")", color = 'yellow')
ff=open('HNMPRD_thre_FATEMHfile1_pred_gdsc_max.txt','r')
pred=[]
for line in ff:
    pred.append(float(line[:-1]))
ff.close()
ff=open('HNMPRD_thre_FATEMHfile1_real_gdsc_max.txt','r')
real=[]
for line in ff:
    real.append(float(line[:-1]))
ff.close()
predicted_probability=np.array(pred)
real_labels=np.array(real)

precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
aupr_score = auc(recall, precision)
plt.plot(recall,precision,label="HNMPRD (AUC="+str(round(aupr_score,3))+")", color = 'g')


predicted_probability=np.loadtxt('DSCPLMF_CV_pred_0.csv',dtype=float, delimiter=",")
real_labels=np.loadtxt('DSCPLMF_CV_real_0.csv',dtype=float, delimiter=",")

precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
aupr_score = auc(recall, precision)
plt.plot(recall,precision,label="DSPLMF (AUC="+str(round(aupr_score,3))+")", color = 'y')

predicted_probability=np.loadtxt('Ref_CV_pred_0.csv',dtype=float, delimiter=",")
real_labels=np.loadtxt('Ref_CV_real_0.csv',dtype=float, delimiter=",")

precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
aupr_score = auc(recall, precision)
plt.plot(recall,precision,label="Ref-DNN (AUC="+str(round(aupr_score,3))+")", color = 'blueviolet')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc=4, fontsize='small')
plt.savefig('plots\AUPR.png',dpi=720)
plt.show()
