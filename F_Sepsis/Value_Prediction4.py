#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 09:33:49 2018

@author: iam
"""

import tensorflow as tf
import sys
import os
import numpy as np
import pickle
from ValueItr_test import main_network

def main():
    
    tf.InteractiveSession()
    logs_path = '/tmp/tensorflow_logs/example/newversion'
    
    # encoder intializer
    learning_rate = 0.0001
    num_steps = 300
    batch_size=10
    n_feature = 131
    n_hidden_1 = 131 # 1st layer number of neurons
    n_hidden_2 = 85 # 2nd layer number of neurons
    display_step=10
    # tf graph encoder  input
    X = tf.placeholder(tf.float32, [None,131])
    #Acc_loss = tf.placeholder(tf.float32)
    
    
    
    
    # tf graph main network input
    
    St=tf.placeholder(tf.float32, [None,159])
    Stp1=tf.placeholder(tf.float32, [None,159])
    R_CT=tf.placeholder(tf.float32, [None])
    time_stamp=tf.placeholder(tf.float32)
    Acc_Loss=tf.placeholder(tf.float32)
    
    
    
    
    
    
    
    
    # Store layers weight & bias
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_feature,n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_feature]))
    }
    biases = {
        'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.zeros([n_hidden_2])),
        'decoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.zeros([n_feature])),
    }


    
    
    
    def encoder(x):
        layer_1= tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2']))
        #self._encoder = layer_2
        return layer_2 #self._encoder
    
    def decoder(y):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(y,weights['decoder_h1']),biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))
        #self._decoder = layer_2
        return layer_2
    
    #construct a model
    encode_in=encoder(X)
    decoded_out = decoder(encode_in)
    
    # Define loss and optimizer for our encoder 
    #loss_op =tf.add(tf.reduce_mean(tf.pow((X - decoded_out), 2)),Acc_loss)
    loss = tf.reduce_mean(tf.pow(tf.subtract(X,decoded_out), 2))
    optimizer =tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
    
    
    V_Itr=main_network(R_CT=R_CT,St=St,Stp1=Stp1,time_stamp=time_stamp,Acc_Loss=Acc_Loss)  
    
    
    # define main network and function value
    
    
    
    init = tf.initialize_all_variables()
    
    data_path = '/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_ProcessedALL5.pickle'
    foldind_path = '/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_FOLDIND.pickle'
    
    DATA = pickle.load( open( data_path, "rb" ) )['DATA_RL']
    FOLD_IND = pickle.load( open( foldind_path, "rb" ) )['FOLD_INDX']
    for fold_index in range(1): #len(FOLD_IND)
        
        DATA_Train = DATA[0:500]#[FOLD_IND[fold_index]['index_train']]
        DATA_Test = DATA[501:600] #[FOLD_IND[fold_index]['index_test']]   
        
        RL_DATA_X=[]
        RL_DATA_COVAR=[]
        RL_DATA_Rewards=[]
        
        RL_DATA=[]
        
        
        # PRE TRAINING 
        for Data_train in DATA_Train:
            RL_DATA_X.append(np.asarray(np.transpose(Data_train['X'])))
            RL_DATA_COVAR.append(np.asarray(np.transpose(Data_train['COVAR'])))
            RL_DATA_Rewards.append(np.sum(np.asarray(np.transpose(Data_train['Rewards'])),1))
       
        #print RL_DATA_X[4].shape
        AE_DATA=RL_DATA_X[0];
        for d_ind in range(1,len(RL_DATA_X)):
            AE_DATA=np.concatenate((RL_DATA_X[d_ind] ,AE_DATA),0)
        #AE_DATA=np.random.rand(500,131)
        
        with tf.Session() as sess:
            
            # Run the initializer
            sess.run(init)
            summary_writer = tf.summary.FileWriter(logs_path)
            import random 
            #print Acc_Loss.shap()
            
            # auto encoder
            #i =0
            #his_encode_loss_valid=[]
            #early_stop = False
            #step = 1
            
            
            #encoded is inactive
#            while(step<(num_steps+1) and not(early_stop)): 
#                random.shuffle(AE_DATA)
#                AE_DATA_Valid=AE_DATA[0:100,:]
#                AE_DATA_Train=AE_DATA[100:500,:]
#                for batch_ind in range(int(AE_DATA_Train.shape[0]/batch_size)):
#                    
#                    #print(sess.run([optimizer,loss],feed_dict={X:bach_X}))
#                    
#                    bach_X=AE_DATA_Train[batch_ind * batch_size : batch_ind * batch_size + batch_size ,:]
#                    _,batch_loss=sess.run([optimizer,loss],feed_dict={X:bach_X})
#                    if(i%display_step==0):
#                        summary=tf.Summary()
#                        summary.value.add(tag='Training_error', simple_value = batch_loss)
#                        summary_writer.add_summary(summary,i)
#                    i = i+1
#                    
#                validation_loss=sess.run([loss],feed_dict={X:AE_DATA_Valid})
#                his_encode_loss_valid.append(validation_loss)
#                if(step>2):
#                    print 'validation loss'
#                    print np.subtract(his_encode_loss_valid[-1],his_encode_loss_valid[-2])
#                if(step>2 and (np.subtract(his_encode_loss_valid[-1],his_encode_loss_valid[-2])< 0.00001)):
#                    print 'eraly stop'
#                    #early_stop=True
#                step+=1
#            summary_writer.flush()
#            
#            
            #Value iteration
            
            sess.run(init)
        
            # DATA Training
            step = 0
            i =0
            
            print len(RL_DATA_X)
            RL_DATA_X_Valid=RL_DATA_X[0:100]
            RL_DATA_COVAR_Valid=RL_DATA_COVAR[0:100]
            RL_DATA_Rewards_Valid=RL_DATA_Rewards[0:100]
            
            RL_DATA_X_Train=RL_DATA_X[100:500]
            RL_DATA_COVAR_Train=RL_DATA_COVAR[100:500]
            RL_DATA_Rewards_Train=RL_DATA_Rewards[100:500]
            
            
            while(step<(num_steps)):
                from sklearn.utils import shuffle
                RL_DATA_X_Train,RL_DATA_COVAR_Train=shuffle(RL_DATA_X_Train,RL_DATA_COVAR_Train)
                batch_size=10
                for bach_ind in range(batch_size):
                    total_error=[0]
                    for d_ind in range((len(RL_DATA_X_Train)/batch_size)*bach_ind,((len(RL_DATA_X_Train)/batch_size)*bach_ind+batch_size-1)):
                        RL_DATA_encoded_X=RL_DATA_X_Train[d_ind]     #no encoder     sess.run(encode_in,feed_dict={X:RL_DATA_X[d_ind]})
                        temp = RL_DATA_COVAR_Train[d_ind]
                        C_ST=np.concatenate([RL_DATA_encoded_X[0:-1,:],temp[0:-1,:]],1) 
                        N_ST=np.concatenate([RL_DATA_encoded_X[1:,:],temp[1:,:]],1) 
                        C_R=np.asarray(RL_DATA_Rewards_Train[d_ind])
                        t_s=(C_R.shape[0])
                        total_error=sess.run([V_Itr.TD_Zero],feed_dict={R_CT:C_R , St:C_ST , Stp1:N_ST , time_stamp:t_s,Acc_Loss:total_error })[0]
                        i = i+1
                    
                    
                    #JUST FOR FEEDING
                    #print d_ind
                    #print len(RL_DATA_X)
                    RL_DATA_encoded_X=RL_DATA_X_Train[d_ind+1]     #no encodersess.run(encode_in,feed_dict={X:RL_DATA_X[d_ind]})
                    temp = RL_DATA_COVAR_Train[d_ind+1]
                    C_ST=np.concatenate([RL_DATA_encoded_X[0:-1,:],temp[0:-1,:]],1) 
                    N_ST=np.concatenate([RL_DATA_encoded_X[1:,:],temp[1:,:]],1) 
                    C_R = np.asarray(RL_DATA_Rewards_Train[d_ind+1])
                    t_s= (C_R.shape[0])
                    
                    befor_weight=sess.run([V_Itr.prediction],feed_dict={R_CT:C_R,St:C_ST,Stp1:N_ST,time_stamp:t_s,Acc_Loss:total_error})
                    total_error,_=sess.run([V_Itr.TD_Zero,V_Itr.optimization],feed_dict={R_CT:C_R , St:C_ST , Stp1:N_ST , time_stamp:t_s,Acc_Loss:total_error })
                    after_weight=sess.run([V_Itr.prediction],feed_dict={R_CT:C_R,St:C_ST,Stp1:N_ST,time_stamp:t_s,Acc_Loss:total_error})
                    #if (np.sum(np.sum(np.subtract(after_weight,befor_weight))))!=0:
                        #print total_error[0]/10
                        
                   
                    
                    
                    
                    
                    
#                    if(i%display_step==0):
#                            summary2=tf.Summary()
#                            summary2.value.add(tag='Value_iteration_Training_error', simple_value = total_error[0]/10)
#                            summary_writer.add_summary(summary2,i)
                    
                
                for v_ind in range(len(RL_DATA_X_Valid)):
                    total_error=[0]
                    RL_DATA_encoded_X=RL_DATA_X_Valid[v_ind]
                    temp = RL_DATA_COVAR_Valid[v_ind]
                    C_ST=np.concatenate([RL_DATA_encoded_X[0:-1,:],temp[0:-1,:]],1) 
                    N_ST=np.concatenate([RL_DATA_encoded_X[1:,:],temp[1:,:]],1) 
                    C_R=np.asarray(RL_DATA_Rewards_Valid[v_ind])
                    t_s=(C_R.shape[0])
                    total_error=sess.run([V_Itr.TD_Zero],feed_dict={R_CT:C_R , St:C_ST , Stp1:N_ST , time_stamp:t_s,Acc_Loss:total_error })[0]
                print 'validation error mean'
                print total_error
                    
                    
                step+=1
                
                    
            
                
                    
if __name__ == '__main__':
    main()