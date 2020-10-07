# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 22:53:36 2020

@author: fahma
"""

import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Pancreas (13)', 'Soft tissue (14)', 'Myeloma (6)', 'Bone (27)', 'Digestive system (26)', 'Neuroblastoma', 'Aero dig tract', 'Leukemia', 'Nervous system', 'Urogenital system', 'Lung SCLC', 'Skin', 'Lymphoma', 'Breast', 'Large intestine', 'Lung', 'Lung NSCLC', 'Thyroid', 'Kidney'
sizes = [13,14,6,27,26,28,42,44,44,60,28,36,25,30,31,6,67,10,18]
explode = (0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0.2,0,0,0,0,0,0,0.2,0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')
def func(pct, sizes):
    s=sum(sizes)
    v=int(pct*s/100)
    return "({:d})".format(v)
fig1, ax1 = plt.subplots()
col=["lightseagreen","seagreen","green","lime","yellow","gold","orange","red","maroon", "firebrick","salmon","mediumvioletred","purple","m","mediumpurple","skyblue","blue","navy","darkslategray"]
wedges, texts, autotexts=ax1.pie(sizes, explode=explode,autopct=lambda pct: func(pct, sizes),textprops=dict(color="w"),shadow=False, colors =col) #labels=labels, #autopct='%1.1f%%',
ax1.legend(wedges, labels,
          title="TISSUE TYPES",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
plt.setp(autotexts, size=7, weight="bold")
ax1.text(-1,1.15,'Leukemia', fontsize=10)
ax1.text(-1.83,0.01,'Urogenital\n  System', fontsize=10)
ax1.text(1,-0.9,' Lung\nNSCLC', fontsize=10)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
