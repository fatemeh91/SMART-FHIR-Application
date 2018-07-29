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
#from Value_plot import plot
#from tensorflow.python.lib.io import file_io
#from subprocess import call


# this version of main loop is used for baysian optimization 

#flags = tf.app.flags
#FLAGS = flags.FLAGS
#flags.DEFINE_string('input_dir', 'input', 'Input Directory.')
#filepath = 'gs://BUCKET_NAME/'

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
    
    tf.reset_default_graph()
    
    tf.InteractiveSession()
    logs_path = '/tmp/tensorflow_logs/example/newversion'
    
    # encoder intializer
    learning_rate = 0.0001
    num_steps = 1000
    batch_number=1
    batch_size=400
    
    
    # tf graph RNN network set up
    
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
    init = tf.global_variables_initializer()
    import os.path
    data_path = '/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_ProcessedALL5.pickle'  #os.path.join(FLAGS.input_dir,'Emory_Sepsis_RL_Data_ProcessedALL5.pickle')
    data_path_Expriation =  '/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_Expiration8.pickle'      #os.path.join(FLAGS.input_dir,'Emory_Sepsis_RL_Data_Expiration8.pickle')
    foldind_path = '/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_FOLDIND.pickle'    #os.path.join(FLAGS.input_dir,'Emory_Sepsis_RL_Data_FOLDIND.pickle')
    
    FOLD_IND =pickle.load( open( foldind_path, "rb" ) )['FOLD_INDX']   #pickle.load(file_io.FileIO( foldind_path, "r" ) )['FOLD_INDX']
    DATA_Exp = pickle.load( open( data_path_Expriation, "rb" ) )['DATA_EXP']  #pickle.load(file_io.FileIO( data_path_Expriation, "r" ) )['DATA_EXP']
    DATA = pickle.load( open( data_path, "rb" ) )['DATA_RL']   #pickle.load(file_io.FileIO( data_path, "r" ) )['DATA_RL']
    mortality_misclassification=0



    for fold_index in range(1): #len(FOLD_IND)
        reorder_data = np.random.permutation(len(DATA))
        DATA=[DATA[i] for i in reorder_data]  
        #RL_DATA_COVAR_Train=[ RL_DATA_COVAR_Train[i] for i in p]  
        #RL_DATA_Rewards_Train=[ RL_DATA_Rewards_Train[i] for i in p]  


        
        DATA_Train = [ DATA[int(i)-1] for i in FOLD_IND[fold_index]['index_train']  ]           #   [FOLD_IND[fold_index]['index_train']]
        #DATA_Test = DATA[4000:5000]#[ DATA[int(i)-1] for i in FOLD_IND[fold_index]['index_test']  ]    #DATA[FOLD_IND[fold_index]['index_test']]   
        
        RL_DATA_X=[]
        RL_DATA_COVAR=[]
        RL_DATA_Rewards=[]
        RL_DATA_Expiration=[]
        
        RL_DATA=[]
        
        
        # PRE TRAINING 
        for Data_train in DATA_Train:
            RL_DATA_X.append(np.asarray(np.transpose(Data_train['X'])))
            RL_DATA_COVAR.append(np.asarray(np.transpose(Data_train['COVAR'])))
            temp_rewards=np.asarray(np.transpose(Data_train['Rewards']))
            a=np.zeros([len(temp_rewards),1])
            for time_rewards_ind in range(len(temp_rewards)):
                a[time_rewards_ind,0]=r1*temp_rewards[time_rewards_ind,0]+r2*temp_rewards[time_rewards_ind,1]+r3*temp_rewards[time_rewards_ind,2]+r4*temp_rewards[time_rewards_ind,3]+r5*temp_rewards[time_rewards_ind,4]+r6*temp_rewards[time_rewards_ind,5]+r7*temp_rewards[time_rewards_ind,6]+r8*temp_rewards[time_rewards_ind,7]
            RL_DATA_Rewards.append(a)
        for data_exp in DATA_Exp:
            RL_DATA_Expiration.append(data_exp['exp'])
       
    
        with tf.Session() as sess:
            sess.run(init)
        
            # DATA Training
            step = 0
            
            # exctract validation set from training
            RL_DATA_X_Valid=RL_DATA_X[0:int(0.2*len(RL_DATA_X))]
            RL_DATA_COVAR_Valid=RL_DATA_COVAR[0:int(0.2*len(RL_DATA_X))]
            RL_DATA_Rewards_Valid=RL_DATA_Rewards[0:int(0.2*len(RL_DATA_X))]
            RL_DATA_Expiration_Valid=RL_DATA_Expiration[0:int(0.2*len(RL_DATA_X))]
        
        
            RL_DATA_X_Train=RL_DATA_X[int(0.2*len(RL_DATA_X)):]
            RL_DATA_COVAR_Train=RL_DATA_COVAR[int(0.2*len(RL_DATA_X)):]
            RL_DATA_Rewards_Train=RL_DATA_Rewards[int(0.2*len(RL_DATA_X)):]
            RL_DATA_Expiration_Train=RL_DATA_Expiration[int(0.2*len(RL_DATA_X)):]
            
            
            
            all_bach_losses_valid=np.zeros([1,1])
            all_bach_losses=np.zeros([batch_number+1,1])
            
            while(step<(num_steps)):
                batchloss=np.zeros([batch_number+1,1])
                p = np.random.permutation(len(RL_DATA_Rewards_Train))
                RL_DATA_X_Train=[ RL_DATA_X_Train[i] for i in p]  
                RL_DATA_COVAR_Train=[ RL_DATA_COVAR_Train[i] for i in p]  
                RL_DATA_Rewards_Train=[ RL_DATA_Rewards_Train[i] for i in p]  
                batch_number=1

                for bach_ind in range(batch_number):
                    total_error_acroos_batch=[0]
                    # for each pateint in bach    
                    for d_ind in range(400):
                        total_error=[0]
                        pcell=[0]
                        py=[0]
                        nl_t=[1.0]
                        
                        RL_DATA_encoded_X=RL_DATA_X_Train[d_ind]     #no encoder     sess.run(encode_in,feed_dict={X:RL_DATA_X[d_ind]})
                        temp = RL_DATA_COVAR_Train[d_ind]
                        C_ST=np.concatenate([RL_DATA_encoded_X[:,:],temp[:,:]],1)
                        C_R=np.asarray(RL_DATA_Rewards_Train[d_ind])
                        t_s=(C_R.shape[0])

                        #print 'patient:'+'\t'+str(d_ind)
                        for St_ind in range((t_s-1)):
                            cs=np.transpose(np.reshape(C_ST[St_ind],[1,C_ST.shape[1]]))
                            ns=np.transpose(np.reshape(C_ST[St_ind+1],[1,C_ST.shape[1]]))
                            rs=np.reshape(C_R[St_ind],[1,1])
                            
                            t_e,hs=sess.run([V_Itr.TD_Zero,V_Itr.predict],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:[0]})
                            total_error=t_e[0][0]
                            pcell=hs[0][0]
                            py=hs[1][0]
                            
                        #compute the last time stamp
                        cs=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                        ns=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                        rs=np.reshape(C_R[-1],[1,1])
                        nl_t=[0.0]    
                        t_e,hs=sess.run([V_Itr.TD_Zero,V_Itr.predict],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:[0]})
                        total_error=t_e[0][0]
                        #print 'pateint:'+'\t'+str(d_ind)+'step'+'\t'+str(step)+'\t'+'total error: '+'\t'+str(total_error)
                        total_error_acroos_batch+=total_error
                        
                    
                    # for the last pateint in batch
                    total_error=[0]
                    pcell=[0]
                    py=[0]
                    nl_t=[1.0]
                    RL_DATA_encoded_X=RL_DATA_X_Train[400]     
                    temp = RL_DATA_COVAR_Train[400]
                    C_ST=np.concatenate([RL_DATA_encoded_X[:,:],temp[:,:]],1)
                    C_R=np.asarray(RL_DATA_Rewards_Train[400])
                    t_s=(C_R.shape[0])
                    
                
                    for St_ind in range((t_s-1)):
                        cs=np.transpose(np.reshape(C_ST[St_ind],[1,C_ST.shape[1]]))
                        ns=np.transpose(np.reshape(C_ST[St_ind+1],[1,C_ST.shape[1]]))
                        rs=np.reshape(C_R[St_ind],[1,1])
                            
                        t_e,hs=sess.run([V_Itr.TD_Zero,V_Itr.predict],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:[0]})
                        total_error=t_e[0][0]
                        pcell=hs[0][0]
                        py=hs[1][0]
                            
                    #compute the last time stamp for the last patient in batch 
                    cs=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                    ns=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                    rs=np.reshape(C_R[-1],[1,1])
                    nl_t=[0.0]    
                    total_error_acroos_batch+=total_error
                    total_error,_=sess.run([V_Itr.TD_Zero,V_Itr.optimization],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:total_error_acroos_batch})
                        
                    
                    
                    print 'step'+'\t'+str(step)+'total-batch+error:'+'\t'+str(total_error[0][0])
                    batchloss[bach_ind,0]=total_error[0][0]
                    
                        
                        
                    i = i+1
                batchloss[batch_number,0]=np.mean(batchloss)
                all_bach_losses=np.concatenate((batchloss,all_bach_losses), axis=1)
               
                """
                now we wanna to compute the validation error over the batch of size 400 and see how it gonna change by increasing the number of epoch


                      
               #getting validtation error during training 
                for bach_ind in range(batch_number):
                    total_error_across_batch=[0]   
                    for d_ind in range(200):
                        total_error=[0]
                        pcell=[0]
                        py=[0]
                        nl_t=[1.0]
                        
                        RL_DATA_encoded_X=RL_DATA_X_Valid[d_ind]     #no encoder     sess.run(encode_in,feed_dict={X:RL_DATA_X[d_ind]})
                        temp=RL_DATA_COVAR_Valid[d_ind]
                        C_ST=np.concatenate([RL_DATA_encoded_X[:,:],temp[:,:]],1)
                        C_R=np.asarray(RL_DATA_Rewards_Valid[d_ind])
                        t_s=(C_R.shape[0])
                        #during time stamp
                        for St_ind in range((t_s-1)):
                            cs=np.transpose(np.reshape(C_ST[St_ind],[1,C_ST.shape[1]]))
                            ns=np.transpose(np.reshape(C_ST[St_ind+1],[1,C_ST.shape[1]]))
                            rs=np.reshape(C_R[St_ind],[1,1])
                            
                            t_e,hs=sess.run([V_Itr.TD_Zero,V_Itr.predict],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:[0]})
                            total_error=t_e[0][0]
                            pcell=hs[0][0]
                            py=hs[1][0]


                        #compute the last time stamp
                        cs=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                        ns=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                        rs=np.reshape(C_R[-1],[1,1])
                        nl_t=[0.0]    
                        t_e,hs=sess.run([V_Itr.TD_Zero,V_Itr.predict],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:[0]})
                        total_error=t_e[0][0]
                        print 'pateint:'+'\t'+str(d_ind)+'step'+'\t'+str(step)+'\t'+'total error: '+'\t'+str(total_error)
                        total_error_across_batch+=total_error #add current patient error to the total batch error
                    te_arr_tem=np.zeros([1,1])
                    te_arr_tem[0,0]=total_error_across_batch
                    all_bach_losses_valid=np.concatenate((te_arr_tem,all_bach_losses_valid), axis=1)

                """


                step+=1
            
            
            #import pandas as pd
            #from subprocess import call
            #df1 = pd.DataFrame(data=np.transpose(np.asarray(all_bach_losses)))
            #file_name='./per1_crossfold_te_1_rewards_epoch_960_sampling_400patient_1batch_job_error_progress.csv'
            #pd.DataFrame.to_csv(df1,file_name)
            #filepath='gs://emory-sepsis-data2/per1_corossfold_te_1_rewards_epoch_960_sampling_400patient_job_error_progress.csv'
            #call(["gsutil","-m","cp",file_name,filepath])
            #total_error_acroos_batch=[0.0]
            #print 'line 324 after training'
            
            #validation_errors=[]
            #all_validation_patient_state_value=[]
            #for v_ind in range(len(RL_DATA_X_Valid)):
                #validation_patient_state_value=[]
                #total_error=[0]
                #pcell=[0]
                #py=[0]
                #nl_t=[1.0]
                        
                #RL_DATA_encoded_X=RL_DATA_X_Valid[v_ind]     #no encoder     sess.run(encode_in,feed_dict={X:RL_DATA_X[d_ind]})
                #temp = RL_DATA_COVAR_Valid[v_ind]
                #C_ST=np.concatenate([RL_DATA_encoded_X[:,:],temp[:,:]],1)
                #C_R=np.asarray(RL_DATA_Rewards_Valid[v_ind])
                #t_s=(C_R.shape[0])
                        
                #for St_ind in range((t_s-1)):
                    #cs=np.transpose(np.reshape(C_ST[St_ind],[1,C_ST.shape[1]]))
                    #ns=np.transpose(np.reshape(C_ST[St_ind+1],[1,C_ST.shape[1]]))
                    #rs=np.reshape(C_R[St_ind],[1,1])
                    #t_e,hs=sess.run([V_Itr.TD_Zero,V_Itr.predict],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:[0]})
                    #pcell=hs[0][0]
                    #total_error=t_e[0][0]
                    #py=hs[1][0]
                    #ny=hs[2][0]
                    #validation_patient_state_value.append(py)
                    
                #compute the last time stamp
                #cs=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                #ns=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                #rs=np.reshape(C_R[-1],[1,1])
                #nl_t=[0.0]    
                #t_e,hs=sess.run([V_Itr.TD_Zero,V_Itr.predict],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:[0]})
                #pcell=hs[0][0]
                #total_error=t_e[0][0]
                #py=hs[1][0]
                #ny=hs[2][0]
                #validation_errors.append(total_error)
                #validation_patient_state_value.append(py)
                #all_validation_patient_state_value.append(np.transpose(np.asarray(validation_patient_state_value)))
            #print 'before saving to file'
            #import pandas as pd
            #from subprocess import call
            #df2 = pd.DataFrame(data=all_bach_losses_valid)
            #file_name='./per1_crossfold_ve_1_rewards_epoch_960_400patient_job_error_progress.csv'
            #pd.DataFrame.to_csv(df2,file_name)
            #filepath='gs://emory-sepsis-data2/per1_corossfold_ve_1_rewards_epoch_960_400patient_job_error_progress.csv'
            #call(["gsutil","-m","cp",file_name,filepath])


            def classification_error(DATA,RL_DATA_Expiration_Valid):
                def feature_extraction(X):
                    data=[]
                    for x in X:
                        
                        a=[]
                        a.append(min(x))
                        a.append(max(x))
                        a.append(np.std(x))
                        a.append(np.mean(x))
                        data.append(a) 
                        
                    return data
                        
                X_train=feature_extraction(DATA[0:int(len(DATA)*0.7)])
                Y_Train=RL_DATA_Expiration_Valid[0:int(len(DATA)*0.7)]
                
                X_test=feature_extraction(DATA[int(len(DATA)*0.7):len(DATA)])
                Y_Test=RL_DATA_Expiration_Valid[int(len(DATA)*0.7):len(DATA)]
                from sklearn import svm
                clf = svm.SVC()
                print X_train[1:5]
                print len(Y_Train)
                print len(X_train)
                print Y_Train[1:5]
                #Y_Train[1]=1
                #Y_Train[12]=1
                #_Train[14]=1
                #Y_Train[8]=1
                #Y_Train[6]=1
                #Y_Train[24]=1
                
                clf.fit(X_train,Y_Train)
                test_prediction=clf.decision_function(X_test)
                tn=0
                tp=0
                fn=0
                fp=0
                #test_prediction=np.sort(test_prediction)
                for val_ind in range(len(Y_Test)):
                    if Y_Test[val_ind]==test_prediction[val_ind]:
                        if test_prediction[val_ind]==1:
                            tp+=1
                        else:
                            tn+=1
                    else:
                        if test_prediction[val_ind]==1:
                            fp+=1
                        else:
                            fn+=1
                #sensitivity=tp/(tp+fn)
                #specifity=tn/(tn+fp)
                #print sensitivity
                #print specifity
                #print 1-specifity
                error=np.float(fp+fn)
                print 'validation_error_for_current_fold'+str(error)
                return error
                
                
                
                
            print 'last line'   
            #mortality_misclassification+=classification_error(all_validation_patient_state_value,RL_DATA_Expiration_Valid)
            #print 'mortality_misclassification'
    return mortality_misclassification
            
            
            
                
                    
if __name__ == '__main__':
    main()
