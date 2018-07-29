#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 09:18:07 2018

@author: iam
"""

import tensorflow as tf
import sys
import os
import numpy as np
import pickle
from LSTM import LSTM_Bolck
import pandas 
from tensorflow.python.lib.io import file_io
from processing import pooling 
            
def main():
    Main_Loop()
    return None

def Main_Loop():

    

    
    
    
    
    # experiment set up
    tf.reset_default_graph()
    
    tf.InteractiveSession()
    
    learning_rate = 0.0001
    num_steps = 30
    batch_number=100
    batch_size=60
    validation_size=6000
    window_size=70
        
    
    # tf graph RNN network input
    R_st=tf.placeholder(tf.float32,[window_size,1])
    CS=tf.placeholder(tf.float32, [window_size,129]) #xt
    time_stamp=tf.placeholder(tf.int32)
    current_state=tf.placeholder(tf.float32)
    hidden_state=tf.placeholder(tf.float32)
    batch_loss=tf.placeholder(tf.float32)
    last_pat=tf.placeholder(tf.float32)
    
    
    
    V_Itr=LSTM_Bolck(Rewards=R_st,St=CS,current_state=current_state,hidden_state=hidden_state,l_pat=last_pat,Batch_Loss=batch_loss)
    
    """
    reading data from storage 
    """
        
    import os.path
    data_path = '/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_ProcessedALL6.pickle'
    data_path_Expriation ='/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_Expiration10.pickle'
    data_path_Rewards='/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_Rewards2.pickle'
    foldind_path = '/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_FOLDIND2.pickle'


    FOLD_IND = pickle.load(open( foldind_path, "r" ) )['FOLD_INDX']
    DATA_Rewards=pickle.load(open( data_path_Rewards, "r" ) )['DATA_RL']
    DATA_Exp = pickle.load(open( data_path_Expriation, "r" ) )['DATA_EXP']
    DATA = pickle.load(open( data_path, "r" ) )['DATA_RL']
    
    
    
    
    for fold_index in range(1): 

        
        DATA_Train=[ DATA[int(i)] for i in FOLD_IND[fold_index]['index_train'] ]   
        DATA_Train_Rewards=[DATA_Rewards[int(i)] for i in FOLD_IND[fold_index]['index_train']  ]
        DATA_Train_Exp=[DATA_Exp[int(i)] for i in FOLD_IND[fold_index]['index_train']]
        
        
        
        DATA_X=[]
        DATA_COVAR=[]
        DATA_Rewards=[]
        RL_DATA_Expiration=[]
        
        RL_DATA=[]
        
        
        # PRE TRAINING 
        for Data_train in DATA_Train:
            DATA_X.append(np.asarray(np.transpose(Data_train['X'])))
            
            DATA_COVAR.append(np.asarray(np.transpose(Data_train['COVAR'])))
         
        
        for Data_train_rewards in DATA_Train_Rewards:
            temp_rewards=np.asarray(np.transpose(Data_train_rewards['Rewards']))
            a=np.zeros([1,len(temp_rewards)])
            for time_rewards_ind in range(len(temp_rewards)):
                a[0,time_rewards_ind]=(-0.03)*temp_rewards[time_rewards_ind,0]+(-0.4)*temp_rewards[time_rewards_ind,1]+(-0.009)*temp_rewards[time_rewards_ind,2]+(-0.06)*temp_rewards[time_rewards_ind,3]+(-0.01)*temp_rewards[time_rewards_ind,4]+(-0.3)*temp_rewards[time_rewards_ind,5]+(-0.5)*temp_rewards[time_rewards_ind,6]
            DATA_Rewards.append(a)
        
        for data_exp in DATA_Train_Exp:
            RL_DATA_Expiration.append(data_exp['exp'])
        
        
        RL_DATA_X=pooling.pooling(DATA_X,window_size,feature_size=103,'mean')
        RL_DATA_COVAR=pooling.pooling(DATA_COVAR,window_size,feature_size=26,'mean')
        RL_DATA_Rewards=pooling.pooling(DATA_Rewards,window_size,feature_size=1,'mean',transpose='yes')
        
        # exploit validation 
        RL_DATA_X_Valid=RL_DATA_X[0:int(0.2*len(DATA_X))]
        RL_DATA_COVAR_Valid=RL_DATA_COVAR[0:int(0.2*len(DATA_X))]
        RL_DATA_Rewards_Valid=RL_DATA_Rewards[0:int(0.2*len(DATA_X))]
        RL_DATA_Expiration_Valid=RL_DATA_Expiration[0:int(0.2*len(DATA_X))]
        
        

        RL_DATA_X_Train=RL_DATA_X[int(0.2*len(DATA_X)):]
        RL_DATA_COVAR_Train=RL_DATA_COVAR[int(0.2*len(DATA_X)):]
        RL_DATA_Rewards_Train=RL_DATA_Rewards[int(0.2*len(DATA_X)):]
        RL_DATA_Expiration_Train=RL_DATA_Expiration[int(0.2*len(DATA_X)):]
        
        """
        Training the models 
        """
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            training_batchloss_step=np.zeros([batch_number,num_steps])
            validation_batchloss_step=np.zeros([batch_number,num_steps])
            step = 0
            while(step<(num_steps)):
                p = np.random.permutation(len(RL_DATA_Rewards_Train))
                RL_DATA_X_Train=[ RL_DATA_X_Train[i] for i in p]  
                RL_DATA_COVAR_Train=[ RL_DATA_COVAR_Train[i] for i in p]  
                RL_DATA_Rewards_Train=[ RL_DATA_Rewards_Train[i] for i in p]
                RL_DATA_Expiration_Train=[ RL_DATA_Expiration_Train[i] for i in p]
                
                for batch_ind in range(batch_number):
                    cur_batch_loss=[0.]   
                    for patient_ind in range(batch_ind*batch_size,batch_ind*batch_size+batch_size-1):
                         
                        last_patient=[0.]                 
                        RL_DATA_encoded_X=RL_DATA_X_Train[patient_ind] 
                        temp=RL_DATA_COVAR_Train[patient_ind]
                        C_R=np.asarray(RL_DATA_Rewards_Train[patient_ind])
                        C_ST=np.concatenate([RL_DATA_encoded_X[:,:],temp[:,:]],1)
                        C_cell=[0]
                        C_y=[0]
                        ts=C_R.shape[0]
                        error,block_input=sess.run([V_Itr.TD_Zero,V_Itr.prediction],feed_dict={R_st:C_R,CS:C_ST,current_state:C_y,hidden_state:C_cell,last_pat:last_patient[0],batch_loss:cur_batch_loss[0]})
                        cur_batch_loss+=error
                    
                    #last pateint 
                    last_patient=[1]                    
                    C_y=[0]
                    C_cell=[0]
                    RL_DATA_encoded_X=RL_DATA_X_Train[patient_ind+1] 
                    temp=RL_DATA_COVAR_Train[patient_ind+1]
                    C_ST=np.reshape((np.concatenate([RL_DATA_encoded_X[:,:],temp[:,:]],1)),[70,129])
                    C_R=np.asarray(RL_DATA_Rewards_Train[patient_ind+1])
                    error,_=sess.run([V_Itr.TD_Zero,V_Itr.optimization],feed_dict={R_st:C_R,CS:C_ST,current_state:C_y,hidden_state:C_cell,last_pat:last_patient[0],batch_loss:cur_batch_loss[0]})
                    cur_batch_loss=error
                    training_batchloss_step[batch_ind,step]=cur_batch_loss;
                    
                step+=1
                print training_batchloss_step

          

if __name__ == '__main__':
    main()