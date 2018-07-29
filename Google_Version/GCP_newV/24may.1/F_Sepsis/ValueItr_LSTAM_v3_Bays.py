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
from tensorflow.python.lib.io import file_io
from subprocess import call


# this version of main loop is used for baysian optimization 

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'input', 'Input Directory.')
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
    
    learning_rate = 0.0001
    num_steps = 500
    batch_number=10
    batch_size=60
    
    
    
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
    init = tf.global_variables_initializer()
    import os.path
    data_path = os.path.join(FLAGS.input_dir,'Emory_Sepsis_RL_Data_ProcessedALL5.pickle')
    data_path_Expriation = os.path.join(FLAGS.input_dir,'Emory_Sepsis_RL_Data_Expiration9.pickle')
    data_path_Rewards=os.path.join(FLAGS.input_dir,'Emory_Sepsis_RL_Data_Rewards.pickle')
    foldind_path = os.path.join(FLAGS.input_dir,'Emory_Sepsis_RL_Data_FOLDIND.pickle')


    #DATA = pickle.load(file_io.FileIO( data_path, "r" ) )['DATA_RL']
    FOLD_IND = pickle.load(file_io.FileIO( foldind_path, "r" ) )['FOLD_INDX']
    DATA_Rewards=pickle.load(file_io.FileIO( data_path_Rewards, "r" ) )['DATA_RL']
    DATA_Exp = pickle.load(file_io.FileIO( data_path_Expriation, "r" ) )['DATA_EXP']
    DATA = pickle.load(file_io.FileIO( data_path, "r" ) )['DATA_RL']
    
    mortality_misclassification=0



    for fold_index in range(1): #len(FOLD_IND)
        
        DATA_Train = [ DATA[int(i)-1] for i in FOLD_IND[fold_index]['index_train']  ]   
        DATA_Train_Rewards=[DATA_Rewards[int(i)-1] for i in FOLD_IND[fold_index]['index_train']  ]
        DATA_Train_Exp=[DATA_Exp[int(i)-1] for i in FOLD_IND[fold_index]['index_train']  ]
        
        
        RL_DATA_X=[]
        RL_DATA_COVAR=[]
        RL_DATA_Rewards=[]
        RL_DATA_Expiration=[]
        
        RL_DATA=[]
        
        
        # PRE TRAINING 
        for Data_train in DATA_Train:
            RL_DATA_X.append(np.asarray(np.transpose(Data_train['X'])))
            RL_DATA_COVAR.append(np.asarray(np.transpose(Data_train['COVAR'])))
        for Data_train_rewards in DATA_Train_Rewards:
            temp_rewards=np.asarray(np.transpose(Data_train_rewards['Rewards']))
            
            a=np.zeros([1,len(temp_rewards)])
            for time_rewards_ind in range(len(temp_rewards)):
                a[0,time_rewards_ind]=r1*temp_rewards[time_rewards_ind,0]+r2*temp_rewards[time_rewards_ind,1]+r3*temp_rewards[time_rewards_ind,2]+r4*temp_rewards[time_rewards_ind,3]+r5*temp_rewards[time_rewards_ind,4]+r6*temp_rewards[time_rewards_ind,5]+r7*temp_rewards[time_rewards_ind,6]+r8*temp_rewards[time_rewards_ind,7]
            RL_DATA_Rewards.append(a)

        for data_exp in DATA_Train_Exp:
            RL_DATA_Expiration.append(data_exp['exp'])
       
       
        with tf.Session() as sess:
            sess.run(init)
        
            # DATA Training
            step = 0
            i =0
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
                
                for batch_ind in range(batch_number):
                    total_error_acroos_batch=[0]
                    # for each pateint in bach    
                    for d_ind in range(batch_ind*batch_size,batch_ind*batch_size+batch_size-1):
                        total_error=[0]
                        pcell=[0]
                        py=[0]
                        nl_t=[1.0]
                        
                        RL_DATA_encoded_X=RL_DATA_X_Train[d_ind]     #no encoder     sess.run(encode_in,feed_dict={X:RL_DATA_X[d_ind]})
                        temp = RL_DATA_COVAR_Train[d_ind]
                        C_ST=np.concatenate([RL_DATA_encoded_X[:,:],temp[:,:]],1)
                        C_R=np.asarray(RL_DATA_Rewards_Train[d_ind])
                        t_s=(C_R.shape[1]) #another bug

                        #print 'patient:'+'\t'+str(d_ind)
                        for St_ind in range((t_s-1)):
                            cs=np.transpose(np.reshape(C_ST[St_ind],[1,C_ST.shape[1]]))
                            ns=np.transpose(np.reshape(C_ST[St_ind+1],[1,C_ST.shape[1]]))
                            rs=np.reshape(C_R[0,St_ind],[1,1])
                            
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
                        print 'pateint:'+'\t'+str(d_ind)+'step'+'\t'+str(step)+'\t'+'batch_ind'+'\t'+str(batch_ind)+'\t'+'total error: '+'\t'+str(total_error)
                        total_error_acroos_batch+=total_error
                        
                    
                    # for the last pateint in batch
                    total_error=[0]
                    pcell=[0]
                    py=[0]
                    nl_t=[1.0]
                    RL_DATA_encoded_X=RL_DATA_X_Train[batch_ind*batch_size+batch_size]     
                    temp = RL_DATA_COVAR_Train[batch_ind*batch_size+batch_size]
                    C_ST=np.concatenate([RL_DATA_encoded_X[:,:],temp[:,:]],1)
                    C_R=np.asarray(RL_DATA_Rewards_Train[batch_ind*batch_size+batch_size])
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
                        
                    
                    
                    #print 'step'+'\t'+str(step)+'total-batch+error:'+'\t'+str(total_error[0][0])
                    batchloss[batch_ind,0]=total_error[0][0]
                    #print '11*1 shape batchlos baraye har step'+str(batchloss.shape)
                        
                        
                    i = i+1
                batchloss[batch_number,0]=np.mean(batchloss)
                #print 'shpe of concatination' 
                #print '11*1_bTCH LOSS'+str(batchloss.shape)
                #print '11*1_ ALL BATCH LOSS'+str(all_bach_losses.shape)
                all_bach_losses=np.concatenate((all_bach_losses,batchloss), axis=1)
               
                """
                now we wanna to compute the validation error over the batch of size 400 and see how it gonna change by increasing the number of epoch

                """      
                #getting validtation error during training 
                validationlosses_all_batch=np.zeros([1,1])
                for bach_ind in range(1):
                    total_error_across_batch=[0]
                    p = np.random.permutation(len(RL_DATA_Rewards_Train))
                    RL_DATA_X_Train=[ RL_DATA_X_Train[i] for i in p]  
                    RL_DATA_COVAR_Train=[ RL_DATA_COVAR_Train[i] for i in p]  
                    RL_DATA_Rewards_Train=[ RL_DATA_Rewards_Train[i] for i in p]

                    for d_ind in range(60):
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
                    
                    validationlosses_all_batch[0,bach_ind]=total_error_across_batch
                    all_bach_losses_valid=np.concatenate((all_bach_losses_valid,validationlosses_all_batch),axis=1)
                    
                step+=1
            
            
            import pandas as pd
            from subprocess import call
            df1 = pd.DataFrame(data=np.transpose(np.asarray(all_bach_losses)))
            file_name='./te_epoch500_steppermutation_batch10_size60_AUCROCjob_mortalrew.csv'
            pd.DataFrame.to_csv(df1,file_name)
            filepath='gs://emory-sepsis-data2/te_epoch500_steppermutation_batch10size60_AUCROCjob_mortalrew.csv'
            call(["gsutil","-m","cp",file_name,filepath])
            total_error_acroos_batch=[0.0]
            

            
        
            df2 = pd.DataFrame(data=all_bach_losses_valid)
            file_name='./ve_epch500_trainining10batchsize60_ROCAUCjob_mortalrew.csv'
            pd.DataFrame.to_csv(df2,file_name)
            filepath='gs://emory-sepsis-data2/ve_epoch500_training10batchsize60_ROCAUCjob_mortalrew.csv'
            call(["gsutil","-m","cp",file_name,filepath])




            """
            state value of validation 
            """
            validation_errors=[]
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
                    t_e,hs=sess.run([V_Itr.TD_Zero,V_Itr.predict],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:[0]})
                    pcell=hs[0][0]
                    total_error=t_e[0][0]
                    py=hs[1][0]
                    ny=hs[2][0]
                    validation_patient_state_value.append(py)
                    
                #compute the last time stamp
                cs=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                ns=np.transpose(np.reshape(C_ST[-1],[1,C_ST.shape[1]]))
                rs=np.reshape(C_R[-1],[1,1])
                nl_t=[0.0]    
                t_e,hs=sess.run([V_Itr.TD_Zero,V_Itr.predict],feed_dict={R_CT:rs, St:cs,Stp1:ns,cellm1:pcell,ym1:py,Nlast_Ts:nl_t,AccLoss:total_error,Batch_Loss:[0]})
                pcell=hs[0][0]
                total_error=t_e[0][0]
                py=hs[1][0]
                ny=hs[2][0]
                validation_errors.append(total_error)
                validation_patient_state_value.append(py)
                validation_patient_state_value.append(RL_DATA_Expiration_Valid[v_ind])
                all_validation_patient_state_value.append(np.transpose(np.asarray(validation_patient_state_value)))
            
            df3 = pd.DataFrame(data=all_validation_patient_state_value)
            file_name='./vst_nonormalizat_epoch500_10batchrandomlyselectedofsize60_mortalrew.csv'
            pd.DataFrame.to_csv(df3,file_name)
            filepath='gs://emory-sepsis-data2/vst_epoch500_trainin_10batchrandomlyselctedofsize60_mortalrew.csv'
            call(["gsutil","-m","cp",file_name,filepath])
            

            def classification_error(DATA,RL_DATA_Expiration_Valid):
                def feature_extraction(X):
                    data=[]
                    for x in X:
                        
                        a=[]
                        #a.append(min(x))
                        #a.append(max(x))
                        #a.append(np.std(x))
                        a.append(np.mean(x))
                        data.append(a) 
                        
                    return data
                        
                Y_Scores=feature_extraction(DATA)
                Y_True=RL_DATA_Expiration_Valid
        
                #X_test=feature_extraction(DATA[int(len(DATA)*0.7):len(DATA)])
                #Y_Test=RL_DATA_Expiration_Valid[int(len(DATA)*0.7):len(DATA)]
                from sklearn.metrics import roc_auc_score
                AUC=roc_auc_score(Y_True, Y_Scores)
                print 'AUC'+str(AUC)
                from sklearn.metrics import average_precision_score
                average_precision = average_precision_score(Y_True, Y_Scores)
                print('Average precision-recall score: {0:0.2f}'.format(average_precision))
                from sklearn.metrics import precision_recall_curve
                precision, recall, _ = precision_recall_curve(Y_True, Y_Scores)
                print 'percision'+str(precision)
                print 'recal'+str(recall)
                """
                savining precision and recall for showing the curve 
                """
                import pandas as pd
                from subprocess import call
                df1 = pd.DataFrame(data=np.asarray(precision))
                file_name='./precision_ech500_600patient_bch10_size60_AUCROCjob_mortalrew.csv'
                pd.DataFrame.to_csv(df1,file_name)
                filepath='gs://emory-sepsis-data2/precision_ech500_600patient_batch10_size60_AUCROCjob_mortalrew.csv'
                call(["gsutil","-m","cp",file_name,filepath])



                import pandas as pd
                from subprocess import call
                df1 = pd.DataFrame(data=np.asarray(recall))
                file_name='./recall_epoch500_600patient_batch10_size60_AUCROCjob_mortalrew.csv'
                pd.DataFrame.to_csv(df1,file_name)
                filepath='gs://emory-sepsis-data2/recall_epch500_600patient_batch10_size60_AUCROCjob_mortalrew.csv'
                call(["gsutil","-m","cp",file_name,filepath])








            
                #clf.fit(X_train,Y_Train)
                #test_prediction=clf.decision_function(X_test)
                #tn=0
                #tp=0
                #fn=0
                #fp=0
                #test_prediction=np.sort(test_prediction)
                #for val_ind in range(len(Y_Test)):
                    #if Y_Test[val_ind]==test_prediction[val_ind]:
                        #if test_prediction[val_ind]==1:
                            #tp+=1
                        #else:
                            #tn+=1
                    #else:
                        #if test_prediction[val_ind]==1:
                            #fp+=1
                        #else:
                            #fn+=1
                #sensitivity=tp/(tp+fn)
                #specifity=tn/(tn+fp)
                #print sensitivity
                #print specifity
                #print 1-specifity
                #error=np.float(fp+fn)
                print 'AUC_Nomortalityorhospiecesrewards_formartalityclassification'+str(AUC)
                return AUC
                
                
                
                
            
            mortality_AUCclassification=classification_error(all_validation_patient_state_value,RL_DATA_Expiration_Valid)
            #print 'mortality_AUCclassification'
    return mortality_AUCclassification
            
            
            
                
                    
if __name__ == '__main__':
    main()
