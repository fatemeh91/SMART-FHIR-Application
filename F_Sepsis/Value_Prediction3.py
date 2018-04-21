#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 08:54:35 2018

@author: fatemeh
"""


import tensorflow as tf
from main_network2 import main_network
from Auto_encoder3 import Auto_Encoder
import sys
import os
import numpy as np
import pickle

def main():
    
    tf.InteractiveSession()
    X = tf.placeholder(tf.float32, [None,None])
    Feature_no = tf.placeholder(tf.int32,[])
    #Accumulative_loss = tf.placeholder(tf.float32, shape=[None])
    
    
    data_path = '/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_ProcessedALL3.pickle'
    foldind_path = '/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_FOLDIND.pickle'
    DATA = pickle.load( open( data_path, "rb" ) )['DATA']
    FOLD_IND = pickle.load( open( foldind_path, "rb" ) )['FOLD_INDX']
    batch_size=10
    
    
    
    max_iteration = 100 ;   
    
    for fold_index in range(1): #len(FOLD_IND)
        DATA_Train = DATA[0:300]#[FOLD_IND[fold_index]['index_train']]
        DATA_Test = DATA[301:400] #[FOLD_IND[fold_index]['index_test']]   
        AE_DATA_X=[]
        AE_DATA_COVAR=[]
        AE_DATA_Rewards=[]
        
        
        # PRE TRAINING 
        for Data_train in DATA_Train:
            AE_DATA_X.append(np.asarray(np.transpose(Data_train['X'])))
            AE_DATA_COVAR.append(np.asarray(np.transpose(Data_train['COVAR'])))
            AE_DATA_Rewards.append(np.asarray(np.transpose(Data_train['Rewards'])))
            
            
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            import random 
            random.shuffle(DATA_Train)
            batch_size=10
            #sess.run(Auto_Encoder_instance.Auto_Encoder())
            for batch_ind in range(int(len(AE_DATA_X))/batch_size):
                print batch_ind
                for x_ind in range( batch_ind*batch_size , batch_ind*batch_size+batch_size):
                    print [sess.run(Auto_Encoder_instance.loss(),feed_dict={X:AE_DATA_X[x_ind],Feature_no:len(AE_DATA_X[0][0])})[0]]
                        
                print(sess.run(Auto_Encoder_instance.Optimizer,feed_dict={X:AE_DATA_X[x_ind+1],Feature_no:len(AE_DATA_X[0][0])}))
                #else:
                 #   for x_ind in range( batch_ind*batch_size , batch_ind*batch_size+batch_size):
                        
                        #print(sess.run(Auto_Encoder_instance.weight_init,Auto_Encoder_instance.biases_init ))
                  #      ACC_Loss =[sess.run(Auto_Encoder_instance.Accumulative_Loss,feed_dict={X:AE_DATA_X[x_ind],Accumulative_loss:ACC_Loss,Feature_no:len(AE_DATA_X[0][0])})[0]]
                        
            # TRAINING 
#            for itr in range(max_iteration):
#                
#                RL_DATA_X=np.asarray(np.transpose(DATA_Train[batch_ind]['X']))
#                RL_DATA_COVAR=np.asarray(np.transpose(DATA_Train[batch_ind]['COVAR']))
#                RL_DATA_Reqrds = np.asarray(np.transpose(DATA_Train[batch_ind]['Rewards']))
#                
#                
#                
#                
#                DATA_ST=np.asarray(np.transpose(DATA_Train[batch_ind]['X'])[0:-1,:])
#                DATA_SP1T=np.asarray(np.transpose(DATA_Train[batch_ind]['X'])[1:,:])
#                batch_size=5
#                
#            
#            
            
            #print 'Test Set'
            #for i in range(10):
             #   lab_test1 = np.array(lab_test[i]).reshape(1,35)
              #  sess.run(Auto_Encoder_instance.encoder,feed_dict={X:[each_patient['X']],Feature_no:(each_patient['X'].size(),2)})
               # print(sess.run([Auto_Encoder_instance.weight_init , Auto_Encoder_instance.biases_init]))
                
    
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