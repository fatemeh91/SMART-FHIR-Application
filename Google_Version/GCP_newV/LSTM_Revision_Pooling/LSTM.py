#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 11:17:59 2018

@author: fatemeh
"""

import tensorflow as tf
import functools 
from tensorflow.python.util import nest
import numpy as np

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




class LSTM_Bolck():
    def __init__(self,Rewards,St,current_state,hidden_state,time_stamp=70,l_pat=0,Batch_Loss=0,Batchsize=5,n_L_B=1,n_features=129,learning_rate=0.0001):
        
        #place holders
        self.St=St
        self.R=Rewards
        self.time_stamp=time_stamp
        #expriment set up
        self.N_B=n_L_B
        self.n_features=n_features
        self.alpha=0.99 #by bays opt
        self.learning_rate = learning_rate
        self.last_patient =l_pat
        self.batch_Loss=Batch_Loss
        self.Batchsize=Batchsize
        self.ym1=current_state
        self.cellm1=hidden_state
        
        #forget unit
        self.weightf_init
        self.biasesf_init
        
        #input unit
        self.weighti_init
        self.biasesi_init
        
        #control unit
        self.weightz_init
        self.biasesz_init
        
        #output unit
        self.weighto_init
        self.biaseso_init
        
        
        #peephole wights
        self.peepholei_init
        self.peepholef_init
        self.peepholeo_init
        
        #recurents weights 
        self.reccurenti_init
        self.reccurentz_init
        self.reccurentf_init
        self.reccurento_init
        
        self.prediction
        self.TD_Zero 
        self.optimization
        self.weight_check
    #input wight     
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def weightf_init(self):
        weights=tf.Variable(tf.random_normal([self.N_B, self.n_features])) # fully connected neural networks 
        return weights
    
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def weighto_init(self):
        weights=tf.Variable(tf.random_normal([self.N_B, self.n_features])) # fully connected neural networks 
        return weights
    
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def weighti_init(self):
        weights=tf.Variable(tf.random_normal([self.N_B, self.n_features])) # fully connected neural networks 
        return weights
    
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def weightz_init(self):
        weights=tf.Variable(tf.random_normal([self.N_B, self.n_features])) # fully connected neural networks 
        return weights


    # recuurent weight 
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def reccurenti_init(self):
        weights = tf.Variable(tf.random_normal([self.N_B,self.N_B])) # fully connected neural networks 
        return weights

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def reccurentz_init(self):
        
        weights = tf.Variable(tf.random_normal([self.N_B,self.N_B])) # fully connected neural networks 
        return weights
    
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def reccurentf_init(self):
        
        weights = tf.Variable(tf.random_normal([self.N_B,self.N_B])) # fully connected neural networks 
        return weights

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def reccurento_init(self):
        
        weights = tf.Variable(tf.random_normal([self.N_B,self.N_B])) # fully connected neural networks 
        return weights
    

    #biases wight    
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def biasesf_init(self):   
        biase=tf.Variable(tf.zeros(self.N_B))
        return biase

    
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def biaseso_init(self):   
        biase=tf.Variable(tf.zeros(self.N_B))
        return biase
    
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def biasesi_init(self):   
        biase=tf.Variable(tf.zeros(self.N_B))
        return biase
    
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def biasesz_init(self):   
        biase=tf.Variable(tf.zeros(self.N_B))
        return biase    
    
    
    
    #peephole wight
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def peepholef_init(self):   
        weight=tf.Variable(tf.random_normal([self.N_B]))
        return weight

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def peepholeo_init(self):   
        weight=tf.Variable(tf.random_normal([self.N_B]))
        return weight

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def peepholei_init(self):   
        weight=tf.Variable(tf.random_normal([self.N_B]))
        return weight
    
    @define_scope
    def weight_check(self):
        return self.peepholef_init
    
    @define_scope
    def prediction(self):
        # block input 
        def block_input(X,Ym1):
            Z_hat=tf.add(tf.add(tf.matmul(self.weightz_init,tf.reshape(tf.transpose(X),[self.n_features,1])),tf.multiply(self.reccurentz_init,Ym1)),self.biasesz_init)
            Z=tf.tanh(Z_hat)
            return Z
        
        # input gate 
        def input_gate(X,Ym1,cellm1):
            I_hat=tf.add(tf.add(tf.add(tf.matmul(self.weighti_init,tf.reshape(tf.transpose(X),[self.n_features,1])),tf.multiply(self.reccurenti_init,Ym1)),tf.multiply(self.peepholei_init,cellm1)),self.biasesi_init)
            I=tf.sigmoid(I_hat)
            return I


        # forget gate
        def forget_gate(X,Ym1,cellm1):
            F_hat=tf.add(tf.add(tf.add(tf.matmul(self.weightf_init,tf.reshape(tf.transpose(X),[self.n_features,1])),tf.multiply(self.reccurentf_init,Ym1)),tf.multiply(self.peepholef_init,cellm1)),self.biasesf_init)
            F=tf.sigmoid(F_hat)
            return F
        
        # cell
        def cell(X,Ym1,cellm1):
            cell=tf.add(tf.multiply(block_input(X,Ym1),input_gate(X,Ym1,cellm1)),tf.multiply(cellm1,forget_gate(X,Ym1,cellm1)))
            return cell
        
        # output gate
        def output_gate(X,Ym1,cellm1):
            O_hat=tf.add(tf.add(tf.add(tf.matmul(self.weighto_init,tf.reshape(tf.transpose(X),[self.n_features,1])),tf.multiply(self.reccurento_init,Ym1)),tf.multiply(self.peepholeo_init,cell(X,Ym1,cellm1))),self.biaseso_init)
            O=tf.sigmoid(O_hat)
            return O
        # bolck output
        def block_output(X,Ym1,cellm1):
            y=tf.multiply(tf.tanh(cell(X,Ym1,cellm1)),output_gate(X,Ym1,cellm1))
            return y
        
        inputs_series = tf.unstack(self.St,self.time_stamp)
        states_series=[]
        hidden_states_series=[]
        for current_input in inputs_series:
            new_hidden_state=cell(current_input,self.ym1,self.cellm1)
            new_state=block_output(current_input,self.ym1,self.cellm1)
            hidden_states_series.append(new_hidden_state)
            states_series.append(new_state)
            self.cellm1=new_hidden_state
            self.Ym1=new_state
    
            
        return states_series

    @define_scope
    def TD_Zero(self):
        Rewards_series =tf.unstack(self.R,self.time_stamp)
        state=0;
        TD_error=0
        for Rewards in Rewards_series:
            if (state<(len(Rewards_series)-1)):
                TD_error+=tf.abs(tf.subtract(tf.add((self.alpha*self.prediction[state+1]),Rewards),self.prediction[state]))
                state+=1
            else:
                TD_error+=tf.abs(tf.add(tf.subtract(Rewards,self.prediction[-1]),self.last_patient*self.batch_Loss))
    
        return TD_error
    
    @define_scope
    def optimization(self):        
        opt=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.TD_Zero)       
        return opt   
        



