#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:39:09 2018

@author: iam
"""

precision=pd.read_csv('/Users/iam/Documents/eICU/Credit_assingment/Training_validation_error/AUC-ROC/100-withmortal/precision_ech100_600patient_batch10_size60_AUCROCjob_mortalrew.csv')
recall=pd.read_csv('/Users/iam/Documents/eICU/Credit_assingment/Training_validation_error/AUC-ROC/100-withmortal/recall_epch100_600patient_batch10_size60_AUCROCjob_mortalrew.csv')
pre=np.asarray(precision)
re=np.asarray(recall)
plt.fill_between(re[:,1], pre[:,1], step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))