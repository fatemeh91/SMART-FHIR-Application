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
#from LSTM import LSTM_Bolck
import pandas 
#from Value_plot import plot
from tensorflow.python.lib.io import file_io
#from subprocess import call



def standardization(Data1,Data2,size):
    #make data in from 479 * features size
    newdata=[]
    Data=np.concatenate([Data1[:,:],Data2[:,:]],1)
    for i in range(Data1.shape[0]):
        newdata.append(np.reshape(np.asarray(Data[i,:]),[1,Data1.shape[1]+Data2.shape[1]]))
    eraly_stop=len(newdata)
    for j in range(eraly_stop,size):
        newdata.append(np.zeros([1,Data1.shape[1]+Data2.shape[1]]))
    
    return eraly_stop,np.reshape(np.asarray(newdata),[size,Data1.shape[1]+Data2.shape[1]])
        
def reward_std(Data1,size):
    #make data in from 479 * features size
    newdata=[]
    Data=np.asarray(Data1)
    for i in range(Data1.shape[0]):
        newdata.append(np.reshape(np.asarray(Data[i,:]),[1,Data1.shape[1]]))
    eraly_stop=len(newdata)
    for j in range(eraly_stop,size):
        newdata.append(np.zeros([1,Data1.shape[1]]))
    
    return np.reshape(np.asarray(newdata),[size,Data1.shape[1]])

def reforming(x,dims):
    new_data=np.zeros([dims[1],dims[0],dims[2]])
    for i in range(dims[1]):#479
        for j in range(dims[0]):#10
            temp=x[j,i,:]
            new_data[i,j,:]=temp
    return new_data
        
            

def main():
    r=np.ones([8])
    Main_Loop(r)
    return None

def Main_Loop(r):
    import numpy as np
    r1=r[0]
    r2=r[1]
    r3=r[2]
    r4=r[3]
    r5=r[4]
    r6=r[5]
    r7=r[6]
    r8=r[7]
    
    
    import os.path
    data_path = '/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Test_ProcessedALL2.pickle'
    data_path_Expriation ='/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Test_Expiration2.pickle'
    data_path_Rewards='/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Test_Rewards2.pickle'
    #foldind_path = '/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_FOLDIND2.pickle'


    #FOLD_IND = pickle.load(open( foldind_path, "r" ) )['FOLD_INDX']
    DATA_Rewards=pickle.load(open( data_path_Rewards, "r" ) )['DATA_RL']
    DATA_Exp = pickle.load(open( data_path_Expriation, "r" ) )['DATA_EXP']
    DATA = pickle.load(open( data_path, "r" ) )['DATA_RL']
    mortality_misclassification=0
    
    # experiment set up
    tf.reset_default_graph()
    
    tf.InteractiveSession()
    logs_path = '/tmp/tensorflow_logs/example/newversion'
    
    learning_rate = 0.00001
    num_steps =300
    pre_num_steps=5
    batch_number=160
    batch_size=10
    validation_size=10
        
    
    # building a graph 
    hidden_unit=128
    n_steps=100 #max step of sequences 
    seq_width =129
    seq_width_rew=1
    alpha=0.99
    
    seq_input = tf.placeholder(tf.float32, [n_steps, batch_size, seq_width])
    seq_rewards = tf.placeholder(tf.float32, [n_steps, batch_size, seq_width_rew])
    early_stop = tf.placeholder(tf.int32, [batch_size])
    
    
    inputs = [tf.reshape(i, (batch_size, seq_width)) for i in tf.split(seq_input, n_steps,0 )]
    Rewards = [tf.reshape(i, (batch_size, seq_width_rew)) for i in tf.split(seq_rewards, n_steps,0 )]
    
    cell = tf.nn.rnn_cell.LSTMCell(hidden_unit,use_peepholes=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, state = tf.nn.dynamic_rnn(cell,seq_input,dtype=tf.float32)
    
    # pretraining 
    #pre_error= tf.reducesume(tf.pow(tf.subtract(outputs[1:],tf.reducesume(tf.subtract(inputs[0:-1],inputs[1:]))),2))# do a pretraining on (st-st+1)
    #pre_opt=tf.train.RMSPropOptimizer(learning_rate).minimize(pre_error)
    
    
    CV=outputs[0:-1]
    NV=outputs[1:]
    error=tf.reduce_sum(tf.reduce_sum(tf.pow(tf.subtract(CV,tf.add(tf.multiply(alpha,NV),Rewards[1:])),2)))    
    opt=tf.train.RMSPropOptimizer(learning_rate).minimize(error)
    
    
    # training 
    
    for fold_index in range(1): 

        
        DATA_Train=[ DATA[int(i)] for i in range(2000) ]   
        DATA_Train_Rewards=[DATA_Rewards[int(i)] for i in range(2000)  ]
        DATA_Train_Exp=[DATA_Exp[int(i)] for i in range(2000)]
        
        
        
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
                a[0,time_rewards_ind]=r1*temp_rewards[time_rewards_ind,0]+r2*temp_rewards[time_rewards_ind,1]+r3*temp_rewards[time_rewards_ind,2]+r4*temp_rewards[time_rewards_ind,3]+r5*temp_rewards[time_rewards_ind,4]+r6*temp_rewards[time_rewards_ind,5]+r7*temp_rewards[time_rewards_ind,6]
            DATA_Rewards.append(a)
        
        for data_exp in DATA_Train_Exp:
            RL_DATA_Expiration.append(data_exp['exp'])
        
        
        
        
        RL_DATA_X_Valid=DATA_X[0:int(0.2*len(DATA_X))]
        RL_DATA_COVAR_Valid=DATA_COVAR[0:int(0.2*len(DATA_X))]
        RL_DATA_Rewards_Valid=DATA_Rewards[0:int(0.2*len(DATA_X))]
        RL_DATA_Expiration_Valid=RL_DATA_Expiration[0:int(0.2*len(DATA_X))]
        
        

        RL_DATA_X_Train=DATA_X[int(0.2*len(DATA_X)):]
        RL_DATA_COVAR_Train=DATA_COVAR[int(0.2*len(DATA_X)):]
        RL_DATA_Rewards_Train=DATA_Rewards[int(0.2*len(DATA_X)):]
        RL_DATA_Expiration_Train=RL_DATA_Expiration[int(0.2*len(DATA_X)):]
        
        # start and run the session 
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            """
            pre_training
            
            step=0
            while(step<pre_num_steps):
                for batch_ind in range(3,batch_number+3):
                    Data_x=RL_DATA_X_Valid[batch_ind*batch_size:batch_ind*batch_size+batch_size]
                    Data_covar=RL_DATA_COVAR_Valid[batch_ind*batch_size:batch_ind*batch_size+batch_size]
                    Data_rewards=RL_DATA_Rewards_Valid[batch_ind*batch_size:batch_ind*batch_size+batch_size]
                    
                    e_s=[]
                    st_Data=[]
                    rt_Data=[]
                    for patient_no in range(batch_size):
                        stop,new_Data=standardization(Data_x[patient_no],Data_covar[patient_no],n_steps)
                        rew_new_Data=reward_std(np.transpose(Data_rewards[patient_no]),n_steps)
                        e_s.append(stop) # list of early stop
                        st_Data.append(new_Data) #list of state values including batch patients and 479 time step 
                        rt_Data.append(rew_new_Data) # list of rewards of batch 
                    
                    
                    st_Data=np.reshape(np.asarray(st_Data),[batch_size,n_steps,129])
                    rt_Data=np.reshape(np.asarray(rt_Data),[batch_size,n_steps,1])
                    st_Data=reforming(st_Data,[batch_size,n_steps,129])
                    rt_Data=reforming(rt_Data,[batch_size,n_steps,1])
                    
                    
                    pre_err,_=validation_error=sess.run([pre_error,pre_opt],feed_dict={early_stop:e_s,seq_input:st_Data,seq_rewards:rt_Data})
                    print pre_err
                    
            """
                
                
            
            """
            TRAINIGN
            """
            training_batchloss_step=np.zeros([batch_number,num_steps])
            validation_batchloss_step=np.zeros([batch_number,num_steps])
            step = 0
            while(step<(num_steps)):
                # restart batch loss per step
                
                
                
                
                #permute training data for sampling 
                p = np.random.permutation(len(RL_DATA_Rewards_Train))
                RL_DATA_X_Train=[ RL_DATA_X_Train[i] for i in p]  
                RL_DATA_COVAR_Train=[ RL_DATA_COVAR_Train[i] for i in p]  
                RL_DATA_Rewards_Train=[ RL_DATA_Rewards_Train[i] for i in p]
                RL_DATA_Expiration_Train=[ RL_DATA_Expiration_Train[i] for i in p]
                
                for batch_ind in range(batch_number):
                    Data_x=RL_DATA_X_Train[batch_ind*batch_size:batch_ind*batch_size+batch_size]
                    Data_covar=RL_DATA_COVAR_Train[batch_ind*batch_size:batch_ind*batch_size+batch_size]
                    Data_rewards=RL_DATA_Rewards_Train[batch_ind*batch_size:batch_ind*batch_size+batch_size]
                    e_s=[]
                    st_Data=[]
                    rt_Data=[]
                    for patient_no in range(batch_size):
                        stop,new_Data=standardization(Data_x[patient_no],Data_covar[patient_no],n_steps)
                        rew_new_Data=reward_std(np.transpose(Data_rewards[patient_no]),n_steps)
                        e_s.append(stop) # list of early stop
                        st_Data.append(new_Data) #list of state values including batch patients and 479 time step 
                        rt_Data.append(rew_new_Data) # list of rewards of batch 
                    
                    
                    st_Data=np.reshape(np.asarray(st_Data),[batch_size,n_steps,129])
                    rt_Data=np.reshape(np.asarray(rt_Data),[batch_size,n_steps,1])
                    st_Data=reforming(st_Data,[batch_size,n_steps,129])
                    rt_Data=reforming(rt_Data,[batch_size,n_steps,1])
                    
                    te=sess.run([initial_state],feed_dict={early_stop:e_s,seq_input:st_Data,seq_rewards:rt_Data})
                    
                    training_error,_=sess.run([error,opt],feed_dict={early_stop:e_s,seq_input:st_Data,seq_rewards:rt_Data})
                    #print training_error
                    training_batchloss_step[batch_ind,step]=training_error
                
                
                
                for batch_ind in range(1):
                    Data_x=RL_DATA_X_Valid[batch_ind*batch_size:batch_ind*batch_size+batch_size]
                    Data_covar=RL_DATA_COVAR_Valid[batch_ind*batch_size:batch_ind*batch_size+batch_size]
                    Data_rewards=RL_DATA_Rewards_Valid[batch_ind*batch_size:batch_ind*batch_size+batch_size]
                    
                    e_s=[]
                    st_Data=[]
                    rt_Data=[]
                    for patient_no in range(batch_size):
                        stop,new_Data=standardization(Data_x[patient_no],Data_covar[patient_no],n_steps)
                        rew_new_Data=reward_std(np.transpose(Data_rewards[patient_no]),n_steps)
                        e_s.append(stop) # list of early stop
                        st_Data.append(new_Data) #list of state values including batch patients and 479 time step 
                        rt_Data.append(rew_new_Data) # list of rewards of batch 
                    
                    
                    st_Data=np.reshape(np.asarray(st_Data),[batch_size,n_steps,129])
                    rt_Data=np.reshape(np.asarray(rt_Data),[batch_size,n_steps,1])
                    st_Data=reforming(st_Data,[batch_size,n_steps,129])
                    rt_Data=reforming(rt_Data,[batch_size,n_steps,1])
                    
                    
                    validation_error=sess.run(error,feed_dict={early_stop:e_s,seq_input:st_Data,seq_rewards:rt_Data})
                    validation_batchloss_step[batch_ind,step]=validation_error
                    
                    
                    
                    
                    
                    
                    
                    
                    
                #print training_batchloss_step        
                #print validation_batchloss_step
                
                step+=1;
            print training_batchloss_step
            print validation_batchloss_step
            import pandas as pd
            df1=pd.DataFrame(training_batchloss_step)
            pd.DataFrame.to_csv(df1,'training_batchloss_300step_1_10learningrate.csv')
            
            df2=pd.DataFrame(validation_batchloss_step)
            pd.DataFrame.to_csv(df2,'validation_batchloss_300step_1_10learningrate.csv')
            
            state_values_estimation=[]
            batch_seq_length=[]
            for batch_ind in range(1,2):
                Data_x=RL_DATA_X_Valid[batch_ind*batch_size:batch_ind*batch_size+batch_size]
                Data_covar=RL_DATA_COVAR_Valid[batch_ind*batch_size:batch_ind*batch_size+batch_size]
                Data_rewards=RL_DATA_Rewards_Valid[batch_ind*batch_size:batch_ind*batch_size+batch_size]
                Data_expiration=RL_DATA_Expiration_Valid[batch_ind*batch_size:batch_ind*batch_size+batch_size]
                e_s=[]
                st_Data=[]
                rt_Data=[]
                for patient_no in range(batch_size):
                    stop,new_Data=standardization(Data_x[patient_no],Data_covar[patient_no],n_steps)
                    rew_new_Data=reward_std(np.transpose(Data_rewards[patient_no]),n_steps)
                    e_s.append(stop) # list of early stop
                    st_Data.append(new_Data) #list of state values including batch $
                    rt_Data.append(rew_new_Data) # list of rewards of batch 
                
                
                st_Data=np.reshape(np.asarray(st_Data),[batch_size,n_steps,129])
                rt_Data=np.reshape(np.asarray(rt_Data),[batch_size,n_steps,1])
                st_Data=reforming(st_Data,[batch_size,n_steps,129])
                rt_Data=reforming(rt_Data,[batch_size,n_steps,1])
                
                
                state_estimation=sess.run(outputs,feed_dict={early_stop:e_s,seq_input:st_Data,seq_rewards:rt_Data})
                state_values_estimation.append(state_estimation)
                batch_seq_length.append(e_s)
        
        
        batch_seq_length=np.reshape(np.asarray(batch_seq_length[0]),[batch_size,1])
        state_estimation=np.reshape(np.asarray(state_values_estimation[0]),[n_steps,batch_size,1])
        
        state_estimation=reforming(state_estimation,[n_steps,batch_size,1])
        
        state_estimation2=np.reshape(np.asarray(state_values_estimation[0]),[n_steps,batch_size])
        
        def classification_error(DATA,DATA_Length,RL_DATA_Expiration_Valid):
            def feature_extraction(X,L):
                data=[]
                for x_ind in range(X.shape[0]):
                    temp=X[x_ind,L[x_ind][0]-1]        #np.max(X[x_ind,0:L[x_ind][0]])
                    data.append(temp) 
                return data
            
            Y_Scores=feature_extraction(DATA,DATA_Length)
            Y_True=RL_DATA_Expiration_Valid
            first_class_num=0
            sec_call_num=0
            for label in Y_True:
                if label==0:
                    first_class_num+=1
                else:
                    sec_call_num+=1
            print first_class_num
            print sec_call_num
            from sklearn.metrics import roc_auc_score
            AUC=roc_auc_score(Y_True, Y_Scores)
            print 'AUC'+str(AUC)
            from sklearn.metrics import average_precision_score
            average_precision = average_precision_score(Y_True, Y_Scores)
            #print('Average precision-recall score: {0:0.2f}'.format(average_precisi$
            from sklearn.metrics import precision_recall_curve
            precision, recall, _ = precision_recall_curve(Y_True, Y_Scores)
            return AUC
        mortality_AUCclassification=classification_error(state_estimation,batch_seq_length,Data_expiration)


                
                
                #print state_estimation
                #print state_estimation.shape
                
        
        
        
        
                
            
            



if __name__ == '__main__':
    main()