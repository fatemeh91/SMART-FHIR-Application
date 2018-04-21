#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:56:42 2018

@author: fatemeh
"""
import tensorflow as tf
import functools 



def doublewrap(function):
    """
    A decorator allowing  us to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator



@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator






class main_network():

    def __init__(self,R_CT,St,Stp1,time_stamp,Acc_Loss,n_featurs=159,n_h=[159,80,1],n_layers=3,n_out=1,gamma = 0.2,alpha=0.99,learning_rate=0.0001):
        self.R = R_CT
        self.St=St
        self.Stp1=Stp1
        self.n_input=n_featurs
        
        
        self.n_h = n_h
        self.TS = time_stamp
        self.Acc_Loss=Acc_Loss
        self.n_layers = n_layers 
        self.n_output = n_out 
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.weight_init
        self.biases_init
        self.TD_Zero 
        self.prediction
        self.optimization
        
        
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def weight_init(self):
        
        weights = []
        l1_weights = tf.Variable(tf.truncated_normal([self.n_input, self.n_h[0]],stddev=0.001)) # fully connected neural networks 
        weights.append(l1_weights)
        layers_no =0 
        while (layers_no < self.n_layers-1) :
            l_weights = tf.Variable(tf.truncated_normal([self.n_h[layers_no], self.n_h[layers_no + 1]],stddev=0.001))
            weights.append(l_weights)
            layers_no +=1
        ## constructing out putlayer
        l_weights = tf.Variable(tf.truncated_normal([self.n_h[layers_no], self.n_output],stddev=0.001))
        weights.append(l_weights)
        return weights

    @define_scope#(initializer=tf.contrib.slim.xavier_initializer())
    def biases_init(self):   
        biase = []
        l1_bias=tf.Variable(tf.zeros([self.n_h[0]]))
        biase.append(l1_bias)
        layers_no =0 
        while (layers_no < self.n_layers-1) :
            l_bias=tf.Variable(tf.zeros([self.n_h[layers_no + 1]]))
            biase.append(l_bias)
            layers_no +=1
        l_bias=tf.Variable(tf.zeros([self.n_output]))
        biase.append(l_bias)
            
        return biase
    
    @define_scope
    def prediction(self):
        out_layer = tf.nn.sigmoid(tf.add(tf.matmul(self.St,self.weight_init[0]),self.biases_init[0]))
        layer_no =1 
        while layer_no< self.n_layers:
            out_layer = tf.nn.sigmoid(tf.add(tf.matmul(out_layer,self.weight_init[layer_no]),self.biases_init[layer_no]))
            layer_no +=1 
        
        max_pooling= tf.reduce_max(out_layer)#tf.nn.pool(out_layer,[1,tf.shape(out_layer)[0]],"AVG","SAME")
        return self.weight_init[0]
    
          
    
#    @define_scope
#    def feed_forward_neural_net(self,X):
#         out_layer = tf.nn.sigmoid(tf.add(tf.matmul(X,self.weight_init[0]),self.biases_init[0]))
#         layer_no =1 
#         while layer_no< self.n_layers:
#            out_layer = tf.nn.sigmoid(tf.add(tf.matmul(out_layer,self.weight_init[layer_no]),self.biases_init[layer_no]))
#            layer_no +=1 
#            
#        
#         return out_layer

    @define_scope
    def TD_Zero(self):
        
        
        
        
        def feed_forward_neural_net(X):
            out_layer = tf.nn.sigmoid(tf.add(tf.matmul(X,self.weight_init[0]),self.biases_init[0]))
            layer_no =1 
            while layer_no< self.n_layers:
                out_layer = tf.nn.sigmoid(tf.add(tf.matmul(out_layer,self.weight_init[layer_no]),self.biases_init[layer_no]))
                layer_no +=1 
            return out_layer
        
        V_CT = feed_forward_neural_net(X=self.St)
        V_NT = feed_forward_neural_net(X=self.Stp1)
        TD_error = tf.subtract(tf.add(tf.scalar_mul(self.alpha,V_NT),self.R[0:-1]),V_CT)
        total_error=tf.reduce_sum(tf.multiply(tf.pow(TD_error,2),(0.5/self.TS)))
        total_error+=tf.multiply(tf.pow(tf.subtract(V_NT[-1],self.R[-1]),2),(0.5/self.TS))
        total_error+=self.Acc_Loss
        
        return total_error
    
    @define_scope
    def optimization(self):
        opt=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.TD_Zero/10)
        return opt        
        
        
    
    
    
    
    

import tensorflow as tf
import sys
import os
import numpy as np
import pickle

def main():
    St=tf.placeholder(tf.float32, [None,159])
    Stp1=tf.placeholder(tf.float32, [None,159])
    R_CT=tf.placeholder(tf.float32, [None])
    time_stamp=tf.placeholder(tf.int32)
    
    V_Itr=main_network(R_CT=R_CT,St=St,Stp1=Stp1,time_stamp=time_stamp)
    print 'fnjkn'
    data_path = '/Users/iam/Documents/eICU/Credit_assingment/Shamim_Value_Iteration/Value_Iteration/Emory_Sepsis_RL_Data_ProcessedALL5.pickle'
    DATA = pickle.load( open( data_path, "rb" ) )['DATA_RL']
    
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
    
    init = tf.initialize_all_variables()        
    with tf.Session() as sess:
        sess.run(init)
        for step in range(30):
            for d_ind in range(len(RL_DATA_X)):
                temp = RL_DATA_COVAR[d_ind]
                temp2=RL_DATA_X[d_ind]
                C_ST=np.concatenate([temp2[0:-1,:],temp[0:-1,:]],1) 
                N_ST=np.concatenate([temp2[1:,:],temp[1:,:]],1) 
                C_R = np.asarray(RL_DATA_Rewards[d_ind])
                t_s= (C_R.shape[0])
                
                befor_weight=sess.run([V_Itr.prediction],feed_dict={R_CT:C_R,St:C_ST,Stp1:N_ST,time_stamp:t_s})
                ccc=sess.run([V_Itr.TD_Zero],feed_dict={R_CT:C_R , St:C_ST , Stp1:N_ST , time_stamp:t_s })
                #print ccc
                after_weight=sess.run([V_Itr.prediction],feed_dict={R_CT:C_R,St:C_ST,Stp1:N_ST,time_stamp:t_s})
                
                #print 'after'
                if np.sum(np.sum(np.subtract(after_weight,befor_weight)))!=0:
                    ff=0
                    print(np.subtract(after_weight,befor_weight))
                
                #print(sess.run([V_Itr.prediction],feed_dict={R_CT:C_R,St:C_ST,Stp1:N_ST,time_stamp:t_s}))

                        
if __name__ == '__main__':
    main()