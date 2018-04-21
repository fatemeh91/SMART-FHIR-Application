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
from LSTM import LSTM_Bolck
import pandas 
from Value_plot import plot

def main():
    Main_Loop(r1=1,r2=1,r3=1,r4=1,r5=1,r6=1,r7=1,r8=1)
    return None


def Main_Loop(r1,r2,r3,r4,r5,r6,r7,r8):
    tf.reset_default_graph()
    
    tf.InteractiveSession()
    logs_path = '/tmp/tensorflow_logs/example/newversion'
    
    # encoder intializer
    learning_rate = 0.0001
    num_steps = 300
    batch_size=40
    
    #auto encoder setup
    n_feature = 131
    n_hidden_1 = 131 # 1st layer number of neurons
    n_hidden_2 = 85 # 2nd layer number of neurons
    X = tf.placeholder(tf.float32, [None,131])
    Acc_loss = tf.placeholder(tf.float32)
    
    

    
    
    # Store layers weight & bias of encoder -decoder
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




    
    
    # tf graph RNN network input
    
    St=tf.placeholder(tf.float32, [159,1])
    Stp1=tf.placeholder(tf.float32, [159,1])
    R_CT=tf.placeholder(tf.float32)
    AccLoss=tf.placeholder(tf.float32)
    ym1=tf.placeholder(tf.float32)
    cellm1=tf.placeholder(tf.float32)
    Nlast_Ts=tf.placeholder(tf.float32)
    Batch_Loss=tf.placeholder(tf.float32)
    V_Itr=LSTM_Bolck(Rewards=R_CT,St=St,Stp1=Stp1,cellm1=cellm1,ym1=ym1,Nlast_Ts=Nlast_Ts,AccLoss=AccLoss,Batch_Loss=Batch_Loss)  
    
    
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
#        AE_DATA=RL_DATA_X[0];
#        for d_ind in range(1,len(RL_DATA_X)):
#            AE_DATA=np.concatenate((RL_DATA_X[d_ind] ,AE_DATA),0)
#        
        with tf.Session() as sess:
            
            
            # Run the initializer
#            sess.run(init)
#            import random 
            
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
            
            RL_DATA_X_Valid=RL_DATA_X[0:100]
            RL_DATA_COVAR_Valid=RL_DATA_COVAR[0:100]
            RL_DATA_Rewards_Valid=RL_DATA_Rewards[0:100]
            
            RL_DATA_X_Train=RL_DATA_X[100:500]
            RL_DATA_COVAR_Train=RL_DATA_COVAR[100:500]
            RL_DATA_Rewards_Train=RL_DATA_Rewards[100:500]
            
            all_bach_losses=np.zeros([41,1])
            while(step<(num_steps)):
                batchloss=np.zeros([41,1])
                from sklearn.utils import shuffle
                #RL_DATA_X_Train,RL_DATA_COVAR_Train=shuffle(RL_DATA_X_Train,RL_DATA_COVAR_Train,RL_DATA_Rewards_Train)
                batch_size=40
                for bach_ind in range(batch_size):
                    total_error_acroos_batch=[0]
                    # for each pateint in bach    
                    for d_ind in range(len(RL_DATA_X_Train)/batch_size*bach_ind,(len(RL_DATA_X_Train)/batch_size*bach_ind)+(len(RL_DATA_X_Train)/batch_size)-1):
                        total_error=[0]
                        pcell=[0]
                        py=[0]
                        nl_t=[1.0]
                        
                        RL_DATA_encoded_X=RL_DATA_X_Train[d_ind]     #no encoder     sess.run(encode_in,feed_dict={X:RL_DATA_X[d_ind]})
                        temp = RL_DATA_COVAR_Train[d_ind]
                        C_ST=np.concatenate([RL_DATA_encoded_X[:,:],temp[:,:]],1)
                        C_R=np.asarray(RL_DATA_Rewards_Train[d_ind])
                        t_s=(C_R.shape[0])
                        
                        for St_ind in range((t_s-1)):
                            cs=np.transpose(np.reshape(C_ST[St_ind],[1,C_ST.shape[1]]))
                            ns=np.transpose(np.reshape(C_ST[St_ind+1],[1,C_ST.shape[1]]))
                            rs=np.reshape(C_R[St_ind],[1,1])
                            
                            t_e,hs=sess.run([V_Itr.TD_Zero,V_Itr.predict],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:total_error_acroos_batch})
                            total_error=t_e[0][0]
                            pcell=hs[0][0]
                            py=hs[1][0]
                            
                        #compute the last time stamp
                        cs=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                        ns=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                        rs=np.reshape(C_R[-1],[1,1])
                        nl_t=[0.0]    
                        total_error=sess.run([V_Itr.TD_Zero],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:total_error_acroos_batch})[0][0]
                        total_error_acroos_batch+=total_error
                        
                    
                    # for the last pateint in batch
                    total_error=[0]
                    pcell=[0]
                    py=[0]
                    nl_t=[1.0]
                    RL_DATA_encoded_X=RL_DATA_X_Train[(len(RL_DATA_X_Train)/batch_size*bach_ind)+(len(RL_DATA_X_Train)/batch_size)-1]     #no encoder     sess.run(encode_in,feed_dict={X:RL_DATA_X[d_ind]})
                    temp = RL_DATA_COVAR_Train[(len(RL_DATA_X_Train)/batch_size*bach_ind)+(len(RL_DATA_X_Train)/batch_size)-1]
                    C_ST=np.concatenate([RL_DATA_encoded_X[:,:],temp[:,:]],1)
                    C_R=np.asarray(RL_DATA_Rewards_Train[(len(RL_DATA_X_Train)/batch_size*bach_ind)+(len(RL_DATA_X_Train)/batch_size)-1])
                    t_s=(C_R.shape[0])
                    
                    for St_ind in range((t_s-1)):
                        cs=np.transpose(np.reshape(C_ST[St_ind],[1,C_ST.shape[1]]))
                        ns=np.transpose(np.reshape(C_ST[St_ind+1],[1,C_ST.shape[1]]))
                        rs=np.reshape(C_R[St_ind],[1,1])
                            
                        t_e,hs=sess.run([V_Itr.TD_Zero,V_Itr.predict],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:total_error_acroos_batch})
                        total_error=t_e[0][0]
                        pcell=hs[0][0]
                        py=hs[1][0]
                            
                    #compute the last time stamp
                    cs=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                    ns=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                    rs=np.reshape(C_R[-1],[1,1])
                    nl_t=[0.0]    
                    total_error_acroos_batch+=total_error
                    total_error,_=sess.run([V_Itr.TD_Zero,V_Itr.optimization],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:total_error_acroos_batch})
                        
                    
                    
                    #print 'error for batch'+'\t'+str(bach_ind)
                    print total_error[0][0]
                    batchloss[bach_ind,0]=total_error[0][0]
                
                    
                        
                        
                    i = i+1
                batchloss[40,0]=np.mean(batchloss)
                all_bach_losses=np.concatenate((batchloss,all_bach_losses), axis=1)
                step+=1
            
            import pandas as pd
            print all_bach_losses.shape
            df1 = pd.DataFrame(data=np.transpose(np.asarray(all_bach_losses)))
            
            pd.DataFrame.to_csv(df1,'./training_error.csv')
            
            # draw a fucntion value for validation prediction and baysian optimization 
            total_error_acroos_batch=[0.0]
            all_validation_patient_state_value=[]
            for v_ind in range(len(RL_DATA_X_Valid)):
                validation_patient_state_value=[]
                total_error=[0]
                pcell=[0]
                py=[0]
                nl_t=[1.0]
                        
                RL_DATA_encoded_X=RL_DATA_X_Valid[v_ind]     #no encoder     sess.run(encode_in,feed_dict={X:RL_DATA_X[d_ind]})
                temp = RL_DATA_COVAR_Valid[v_ind]
                C_ST=np.concatenate([RL_DATA_encoded_X[:,:],temp[:,:]],1)
                C_R=np.asarray(RL_DATA_Rewards_Valid[v_ind])
                t_s=(C_R.shape[0])
                        
                for St_ind in range((t_s-1)):
                    cs=np.transpose(np.reshape(C_ST[St_ind],[1,C_ST.shape[1]]))
                    ns=np.transpose(np.reshape(C_ST[St_ind+1],[1,C_ST.shape[1]]))
                    rs=np.reshape(C_R[St_ind],[1,1])
                    hs=sess.run([V_Itr.predict],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:total_error_acroos_batch})
                    pcell=hs[0][0]
                    py=hs[1][0]
                    ny=hs[2][0]
                    validation_patient_state_value.append(py)
                    
                #compute the last time stamp
                cs=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                ns=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                rs=np.reshape(C_R[-1],[1,1])
                nl_t=[0.0]    
                hs=sess.run([V_Itr.V_Itr.predict],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:total_error_acroos_batch})
                pcell=hs[0][0]
                py=hs[1][0]
                ny=hs[2][0]
                validation_patient_state_value.append(py)
                all_validation_patient_state_value.append(validation_patient_state_value)
                
                plot(data=all_validation_patient_state_value)

            
            
            
                
                    
if __name__ == '__main__':
    main()