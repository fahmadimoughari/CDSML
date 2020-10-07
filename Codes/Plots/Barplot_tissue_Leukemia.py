# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 00:51:08 2020

@author: fahma
"""

import matplotlib.pyplot as plt

barWidth = 0.3

ADRML=[0.9096,0.9298,0.8369,0.8701]
HNMPRD=[0.5859,0.6249,0.622,0.7562]
RefDNN=[0.85,0.819,0.779,0.739]
DSPLMF=[0.5488,0.5366,0.5104,0.6691]

GNB=[0.8649,0.8926,0.7784,0.7977]
LR=[0.9017,0.9273,0.8247,0.8546]
RF=[0.9067,0.9317,0.8304,0.8586]
MLP=[0.8963,0.9197,0.7855,0.8347]
Ada=[0.9078,0.9345,0.8252,0.8539]
KNN=[0.9015,0.9282,0.8279,0.8551]
SVM=[ 0.886, 0.9116, 0.8151, 0.8519]

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
plt.bar(r11, SVM, width = barWidth, color = 'mediumorchid', edgecolor = 'black', capsize=7, label='SVM')


plt.xticks([r + 4*barWidth for r in r1], ['AUC', 'AUPR', 'Accuracy', 'F-measure'], fontweight='light')
plt.legend(fontsize='x-small',loc='lower right')
label=['0.909','0.93','0.837','0.87','0.586','0.625','0.622','0.756','0.85','0.819','0.779','0.739','0.549','0.537','0.51','0.669','0.865','0.893','0.778','0.798','0.902','0.927','0.825','0.855','0.907','0.932','0.83','0.859','0.896','0.92','0.785','0.835','0.908','0.934','0.825','0.854','0.901','0.928','0.828','0.855','0.886', '0.911', '0.815', '0.851']
# Text on the top of each barplot
for i in range(len(rs)):
    plt.text(x = rs[i]-0.09 , y = bars[i]+0.04, s = label[i], fontweight='light', size = 6,rotation=90)

plt.ylim((0,1.1))
plt.subplots_adjust(bottom= 0.2, top = 0.98)
plt.savefig('Leukemia.png',dpi=720)
plt.show()
