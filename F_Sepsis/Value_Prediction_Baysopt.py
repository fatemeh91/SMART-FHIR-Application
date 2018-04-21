#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:08:39 2018

@author: iam
"""

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
    Main_Loop(learning_rate = 0.0001,
num_steps = 300,
    batch_size=10,
    n_feature = 131,
    n_hidden_1 = 131,
    n_hidden_2 = 85)
    return None

def Main_Loop(n_hidden_2,alpha):
    
    tf.InteractiveSession()
    logs_path = '/tmp/tensorflow_logs/example/newversion'
    
    learning_rate = 0.0001
    num_steps = 300
    batch_size=10
    n_feature = 131
    n_hidden_1 = 131 # 1st layer number of neurons
    
    display_step=10
    # tf graph encoder  input
    X = tf.placeholder(tf.float32, [None,131])
    #Acc_loss = tf.placeholder(tf.float32)
    
    # tf graph main network input
    St=tf.placeholder(tf.float32, [None,n_hidden_2+28])
    Stp1=tf.placeholder(tf.float32, [None,n_hidden_2+28])
    R_CT=tf.placeholder(tf.float32, [None])
    time_stamp=tf.placeholder(tf.float32)
    
    
    
    
    
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
    
    
    V_Itr=main_network(R_CT=R_CT,St=St,Stp1=Stp1,time_stamp=time_stamp,n_featurs=(n_hidden_2+28))  
    
    
    # define main network and function value
    
    
    
    init = tf.initialize_all_variables()
    
    data_path = '/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_ProcessedALL5.pickle'
    data_path_Expriation = '/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_Expiration.pickle'
    foldind_path = '/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_FOLDIND.pickle'
    
    DATA = pickle.load( open( data_path, "rb" ) )['DATA_RL']
    DATA_Exp= pickle.load( open( data_path_Expriation, "rb" ) )['DATA_EXP']
    FOLD_IND = pickle.load( open( foldind_path, "rb" ) )['FOLD_INDX']
    for fold_index in range(1): #len(FOLD_IND)
        DATA_Train = DATA[0:500]#[FOLD_IND[fold_index]['index_train']]
        DATA_Test = DATA[501:600] #[FOLD_IND[fold_index]['index_test']]   
        
        RL_DATA_X=[]
        RL_DATA_COVAR=[]
        RL_DATA_Rewards=[]
        RL_DATA_Expiration=[]
        
        RL_DATA=[]
        
        
        # PRE TRAINING 
        for Data_train in DATA_Train:
            RL_DATA_X.append(np.asarray(np.transpose(Data_train['X'])))
            RL_DATA_COVAR.append(np.asarray(np.transpose(Data_train['COVAR'])))
            Rewards = np.asarray(np.transpose(Data_train['Rewards']))
            sum_rewards = 0 ; 
            for reward_ind in range(len(Rewards)):
                sum_rewards +=alpha[reward_ind]*reward[reward_ind]
            RL_DATA_Rewards.append(sum_rewards)
        for data_exp in DATA_Exp:
            RL_DATA_Expiration.append(data_exp['exp'])
            
            
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
            i =0
            # auto encoder
            his_encode_loss_valid=[]
            early_stop = False
            step = 1
            while(step<(num_steps+1) and not(early_stop)): 
                random.shuffle(AE_DATA)
                AE_DATA_Valid=AE_DATA[0:100,:]
                AE_DATA_Train=AE_DATA[100:500,:]
                for batch_ind in range(int(AE_DATA_Train.shape[0]/batch_size)):
                    
                    #print(sess.run([optimizer,loss],feed_dict={X:bach_X}))
                    
                    bach_X=AE_DATA_Train[batch_ind * batch_size : batch_ind * batch_size + batch_size ,:]
                    _,batch_loss=sess.run([optimizer,loss],feed_dict={X:bach_X})
                    if(i%display_step==0):
                        summary=tf.Summary()
                        summary.value.add(tag='myVar', simple_value = batch_loss)
                        summary_writer.add_summary(summary,i)
                    i = i+1
                    
                validation_loss=sess.run([loss],feed_dict={X:AE_DATA_Valid})
                his_encode_loss_valid.append(validation_loss)
#                if(step>2):
#                    print np.subtract(his_encode_loss_valid[-1],his_encode_loss_valid[-2])
                if(step>2 and (np.subtract(his_encode_loss_valid[-1],his_encode_loss_valid[-2])< 0.00001)):
                    early_stop=True
                step+=1
            summary_writer.flush()
            
            
            #Value iteration
            sess.run(init)
        
            # DATA Training
            step =0
            num_steps = 30
            early_stop =False
            while(step<num_steps): 
                for d_ind in range(len(RL_DATA_X)):
                    RL_DATA_encoded_X=sess.run(encode_in,feed_dict={X:RL_DATA_X[d_ind]})
                    temp = RL_DATA_COVAR[d_ind]
                    C_ST=np.concatenate([RL_DATA_encoded_X[0:-1,:],temp[0:-1,:]],1) 
                    N_ST=np.concatenate([RL_DATA_encoded_X[1:,:],temp[1:,:]],1) 
                    
                    C_R = np.asarray(RL_DATA_Rewards[d_ind])
                    t_s= (C_R.shape[0])
                    
                    #befor_weight=sess.run([V_Itr.prediction],feed_dict={R_CT:C_R,St:C_ST,Stp1:N_ST,time_stamp:t_s})
                    ccc=sess.run([V_Itr.TD_Zero],feed_dict={R_CT:C_R , St:C_ST , Stp1:N_ST , time_stamp:t_s })
                step+=1
                    
                    #print ccc
                    #after_weight=sess.run([V_Itr.prediction],feed_dict={R_CT:C_R,St:C_ST,Stp1:N_ST,time_stamp:t_s})
                    
                    #print 'after'
                    #if np.sum(np.sum(np.subtract(after_weight,befor_weight)))!=0:
                        #print(np.subtract(after_weight,befor_weight))
                        
                
                    #print(sess.run([wieght_checking],feed_dict={R_CT:C_RT,St:C_ST,Stp1:N_ST,time_stamp:t_s}))
            X_init =[]
            for d_ind in range(len(RL_DATA_X)):
                temp =sess.run(encode_in,feed_dict={X:RL_DATA_X[d_ind]})
                temp2=RL_DATA_COVAR[d_ind]
                x_init=np.concatenate([temp,temp2],1)
                X_init.append(x_init) 
                
            Y_init=0
            for x_inint_ind in range(len(X_init)):
                y_inint=sess.run([V_Itr.prediction],feed_dict={R_CT:C_R , St:X_init[x_inint_ind] , Stp1:X_init[x_inint_ind] , time_stamp:t_s })
            if (y_inint > 0.5 and RL_DATA_Expiration[x_inint_ind]) or (y_inint < 0.5 and not(RL_DATA_Expiration[x_inint_ind])) :
                Y_init +=-1
            else:
                Y_init +=2
    
    return Y_init 
              
            
                
                    
if __name__ == '__main__':
    main()