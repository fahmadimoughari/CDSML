# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 05:20:15 2020

@author: fahma
"""

import matplotlib.pyplot as plt

# width of the bars
barWidth = 0.3

ADRML=[0.9186,0.9349,0.8474,0.8704]
HNMPRD=[0.5722,0.5952,0.6027,0.7353]
RefDNN=[0.896,0.87,0.831,0.805]
DSPLMF=[0.5678,0.5114,0.4872,0.6359]
GNB=[0.8773,0.8911,0.7922,0.8075]
LR=[0.9063,0.9244,0.8388,0.8615]
RF=[0.9152,0.9346,0.8422,0.8613]
MLP=[0.9067,0.9244,0.825,0.8454]
Ada=[0.9164,0.9349,0.8406,0.8621]
KNN=[0.9094,0.9292,0.8412,0.8607]
SVM=[0.9057, 0.9246, 0.8391, 0.8619]

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
label=['0.919','0.935','0.848','0.87','0.572','0.595','0.603','0.735','0.896','0.87','0.831','0.805','0.568','0.511','0.487','0.636','0.877','0.891','0.792','0.807','0.906','0.924','0.839','0.861','0.915','0.935','0.842','0.861','0.907','0.924','0.825','0.845','0.916','0.935','0.84','0.862','0.909','0.929','0.841','0.86', '0.905', '0.925', '0.839', '0.861']

# Text on the top of each barplot
for i in range(len(rs)):
    plt.text(x = rs[i]-0.09 , y = bars[i]+0.04, s = label[i], fontweight='light', size = 6,rotation=90)

plt.ylim((0,1.1))
plt.subplots_adjust(bottom= 0.2, top = 0.98)

# Show graphic
plt.savefig('urological.png',dpi=720)
plt.show()
