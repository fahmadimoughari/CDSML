# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 05:39:59 2020

@author: fahma
"""


import matplotlib.pyplot as plt

# width of the bars
barWidth = 0.3
ADRML=[0.9106,0.9343,0.8358,0.8631]
HNMPRD=[0.5711,0.6035,0.615,0.7487]
RefDNN=[0.874,0.841,0.784,0.758]
DSPLMF=[0.5647,0.5465,0.5084,0.6655]
GNB=[0.8652,0.8914,0.7817,0.8003]
LR=[0.8975,0.9222,0.8203,0.8514]
RF=[0.9087,0.9348,0.8318,0.8566]
MLP=[0.895,0.9184,0.8087,0.8366]
Ada=[0.908,0.9352,0.8276,0.8549]
KNN=[0.8981,0.9259,0.8203,0.8471]
SVM=[0.8965, 0.9216, 0.8189, 0.8503]

# The x position of bars
r1 = [1,5,9,13]
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]
r8 = [x + barWidth for x in r7] 
r9 = [x + barWidth for x in r8] 
r10 = [x + barWidth for x in r9] 
r11 = [x + barWidth for x in r10] 
rs=r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11

bars=ADRML+HNMPRD+RefDNN+DSPLMF+GNB+LR+RF+MLP+Ada+KNN+SVM

# Create the bars
plt.bar(r1, ADRML, width = barWidth, color = 'navy', edgecolor = 'black', capsize=7, label='CDSML')
plt.bar(r2, HNMPRD, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='HNMPRD')
plt.bar(r3, RefDNN, width = barWidth, color = 'skyblue', edgecolor = 'black', capsize=7, label='RefDNN')
plt.bar(r4, DSPLMF, width = barWidth, color = 'limegreen', edgecolor = 'black', capsize=7, label='DSPLMF')
plt.bar(r5, GNB, width = barWidth, color = 'olive', edgecolor = 'black', capsize=7, label='GNB')
plt.bar(r6, LR, width = barWidth, color = 'y', edgecolor = 'black', capsize=7, label='LR')
plt.bar(r7, RF, width = barWidth, color = 'yellow', edgecolor = 'black', capsize=7, label='RF')
plt.bar(r8, MLP, width = barWidth, color = 'orange', edgecolor = 'black', capsize=7, label='MLP')
plt.bar(r9, Ada, width = barWidth, color = 'orangered', edgecolor = 'black', capsize=7, label='ADA')
plt.bar(r10, KNN, width = barWidth, color = 'magenta', edgecolor = 'black', capsize=7, label='KNN')
plt.bar(r11, KNN, width = barWidth, color = 'mediumorchid', edgecolor = 'black', capsize=7, label='SVM')

plt.xticks([r + 4*barWidth for r in r1], ['AUC', 'AUPR', 'Accuracy', 'F-measure'], fontweight='light')

plt.legend(fontsize='x-small',loc='lower right')
label=['0.91','0.934','0.835','0.863','0.571','0.603','0.615','0.748','0.874','0.841','0.784','0.758','0.564','0.546','0.508','0.665','0.865','0.891','0.781','0.80','0.897','0.922','0.82','0.851','0.908','0.934','0.831','0.856','0.895','0.918','0.808','0.836','0.908','0.935','0.827','0.8549','0.898','0.925','0.82','0.847','0.896', '0.921', '0.818', '0.850']

# Text on the top of each barplot
print(len(rs), len(label))
for i in range(len(rs)):
    plt.text(x = rs[i]-0.09 , y = bars[i]+0.04, s = label[i], fontweight='light', size = 6,rotation=90)

plt.ylim((0,1.1))
plt.subplots_adjust(bottom= 0.2, top = 0.98)

# Show graphic
plt.savefig('NSCL.png',dpi=720)
plt.show()
