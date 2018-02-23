#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 08:54:35 2018

@author: fatemeh
"""


import tensorflow as tf
from main_network import main_network
from Auto_Encoder import Auto_Encoder
from FloatOrNone import  FloatOrNone
import sys
import os
import numpy as np

def main():
    tf.InteractiveSession()
    data_path = '/Users/iam/Documents/eICU/Credit_assingment/Value_Iteration/F_Sepsis/my_random_generated.csv' 
    patients = tf.Variable([],name='pateint')
    labdate = tf.Variable([],name = 'labdate')
    X = tf.placeholder(tf.float32, [None,None])
    Feature_no = tf.placeholder(tf.int32,[])
    
    Auto_Encoder_instance=Auto_Encoder(X=X,n_featurs=35,opt = 'Adam')
    Auto_Encoder_instance.Loss_opt()
    
    
    patient_name = []
    lab_date = []
    lab_test = []
        
        
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        with open(data_path) as inf:
            headerline = inf.next()
            header = headerline.strip().split(",")
            for line in inf:
                patient_name.append( str(line.strip().split(",")[0]))
                lab_date.append( str (line.strip().split(",")[1]))
                labtestele = np.asarray(line.strip().split(",")[16:])
                b = []
                for labtest_elements in labtestele:
                    a=FloatOrNone(labtest_elements)
                    b.append(a)
                lab_test.append(np.array(b))
        
        lab_test1 = np.array(lab_test[1]).reshape(1,35)
        labdate = tf.convert_to_tensor(lab_date)
        
        
        print lab_test1.shape[1]
        print lab_test1
        #sess.run(Auto_Encoder_instance.Loss_opt,feed_dict={X:lab_test1,Feature_no:35})
        g=sess.run(Auto_Encoder_instance.encoder,feed_dict={X:lab_test1,Feature_no:35})
        print g
        
#        #sess.run(Auto_Encoder_instance.Auto_Encoder(),)
#        #Auto_Encoder_instance = Auto_Encoder(X,lab_test.shape[1])
#        
#        _, l = sess.run(Loss, feed_dict={X:lab_test[1]})
#        print l.eval()
#    #main_nn = main_network(num_features,n_h1,n_h2)
#    #Value_Iteration(main_nn,patients,labtest,labdate)# for now is just td 1 to build the sckolten 
#     
            
    
    return None

def Value_Iteration(main_nn,patients,labtest,labdate):
    for patient_num in range(patients.get_shape()[0]):
        for i in range(labdate.get_shape()[0]): 
            X = labtest[patient_num,i] # current statet
            R = rewards[patient_num,i] # immidiate reward
            main_nn.X = X
            
            
            
       








#it's not be main function
if __name__ == '__main__':
    main()