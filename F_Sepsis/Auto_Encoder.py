#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:10:56 2018

@author: fatemeh
"""

# deep auto encoder with 2 hiiden layer 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import functools 


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
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
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
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




class Auto_Encoder():

    def __init__(self,X,n_featurs=35,n_h=[128,70],learning_rate = 0.0,beta1=0.9,beta2=0.999,epsilon=1e-08,opt='Adam'):
    
        self.X = X  # # fo each patients at each time step
        self.n_h = n_h
        self.n_input = n_featurs
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.opt = opt
        
        self.weight_init 
        self.biases_init
        self.encoder  
        self.decoder  
        self.Loss_opt 
        
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def weight_init(self):
        
        #if not self._weight_init:
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_h[0]] )),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_h[0],self.n_h[1]])),
            'decoder_h1': tf.Variable(tf.random_normal([self.n_h[1], self.n_h[0]])),
            'decoder_h2': tf.Variable(tf.random_normal([self.n_h[0],self.n_input])),
                }   
        #self._weight_init = weights
        return weights #self._weight_init
    
    @define_scope
    def biases_init(self):
        
        
        #if not self._biases_init:
        biases = {
                'encoder_b1': tf.Variable(tf.random_normal([self.n_h[0]])),
                'encoder_b2': tf.Variable(tf.random_normal([self.n_h[1]])),
                'decoder_b1': tf.Variable(tf.random_normal([self.n_h[0]])),
                'decoder_b2': tf.Variable(tf.random_normal([self.n_input])),
                }
        #self._biases_init = biases
        return biases #self._biases_init
        
        
    @define_scope   
    # Building the encoder
    def encoder(self):
        #if not self._encoder:
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.X, self.weight_init['encoder_h1']),
                                       self.biases_init['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weight_init['encoder_h2']),
                                       self.biases_init['encoder_b2']))
        #self._encoder = layer_2
        return layer_2 #self._encoder
    
    @define_scope  
    # Building the decoder
    def decoder(self):
        #if not self._decoder:
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.encoder, self.weight_init['decoder_h1']),
                                       self.biases_init['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weight_init['decoder_h2']),
                                       self.biases_init['decoder_b2']))
        #self._decoder = layer_2
        return layer_2 #self._decoder
    
    @define_scope  
    def Loss_opt(self):
        loss = tf.reduce_mean(tf.pow(tf.subtract(self.X, self.decoder), 2))
        if self.opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=self.beta1,beta2=self.beta2,epsilon=self.epsilon,use_locking=False).minimize(loss)
        elif self.opt=='RMS':    
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)
        #self._Loss_opt = optimizer
        return optimizer
    
    
    