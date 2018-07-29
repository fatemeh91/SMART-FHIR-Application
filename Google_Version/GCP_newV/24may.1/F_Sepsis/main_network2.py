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






class main_network(object):

    def __init__(self,R_CT,St,Stp1,time_stamp,n_featurs=113,n_h=[113,65,1],n_layers=3,n_out=1,gamma = 0.2,alpha=0.1,learning_rate=0.0001):
        self.R = R_CT
        self.St=St
        self.Stp1=Stp1
        self.n_input=n_featurs
        
        
        self.n_h = n_h
        self.TS = time_stamp
        self.n_layers = n_layers 
        self.n_output = n_out 
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.weight_init
        self.biases_init
        self.TD_Zero 
        self.prediction
        
        
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def weight_init(self):
        
        weights = []
        l1_weights = tf.Variable(tf.random_normal([self.n_input, self.n_h[0]])) # fully connected neural networks 
        weights.append(l1_weights)
        layers_no =0 
        while (layers_no < self.n_layers-1) :
            l_weights = tf.Variable(tf.random_normal([self.n_h[layers_no], self.n_h[layers_no + 1]]))
            weights.append(l_weights)
            layers_no +=1
        ## constructing out putlayer
        l_weights = tf.Variable(tf.random_normal([self.n_h[layers_no], self.n_output]))
        weights.append(l_weights)
        return weights

    @define_scope#(initializer=tf.contrib.slim.xavier_initializer())
    def biases_init(self):   
        biase = []
        l1_bias=tf.Variable(tf.random_normal([self.n_h[0]]))
        biase.append(l1_bias)
        layers_no =0 
        while (layers_no < self.n_layers-1) :
            l_bias=tf.Variable(tf.random_normal([self.n_h[layers_no + 1]]))
            biase.append(l_bias)
            layers_no +=1
        l_bias=tf.Variable(tf.random_normal([self.n_output]))
        biase.append(l_bias)
            
        return biase
    
    @define_scope
    def prediction(self):
        out_layer = tf.nn.sigmoid(tf.add(tf.matmul(self.St,self.weight_init[0]),self.biases_init[0]))
        layer_no =1 
        while layer_no< self.n_layers:
            out_layer = tf.nn.sigmoid(tf.add(tf.matmul(out_layer,self.weight_init[layer_no]),self.biases_init[layer_no]))
            layer_no +=1 
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
        #time_stamps = tf.map_fn(lambda x:x,self.TS, dtype=tf.int64)
        for t in [tf.map_fn(lambda x:x,self.TS, dtype=tf.int32)]:
            tf.train.RMSPropOptimizer(self.learning_rate).minimize(TD_error[t])
        
        total_error = tf.reduce_sum(TD_error)
        
        return total_error
            
    
    
    
    
    

    