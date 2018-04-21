#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:10:56 2018

@author: fatemeh
"""

# deep auto encoder with 2 hiiden layer 

import tensorflow as tf
import numpy as np
import functools 



    

#def doublewrap(function):
#    """
#    A decorator allowing  us to use the decorator to be used without
#    parentheses if not arguments are provided. All arguments must be optional.
#    """
#    @functools.wraps(function)
#    def decorator(*args, **kwargs):
#        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
#            return function(args[0])
#        else:
#            return lambda wrapee: function(wrapee, *args, **kwargs)
#    return decorator



#@doublewrap
#def define_scope(function, scope=None, *args, **kwargs):
#    """
#    A decorator for functions that define TensorFlow operations. The wrapped
#    function will only be executed once.
#    """
#    attribute = '_cache_' + function.__name__
#    name = scope or function.__name__
#    @property
#    @functools.wraps(function)
#    def decorator(self):
#        if not hasattr(self, attribute):
#            with tf.variable_scope(name, *args, **kwargs):
#                setattr(self, attribute, function(self))
#        return getattr(self, attribute)
#    return decorator




class Auto_Encoder(object):
    
    def __init__(self,X,n_featurs=131,n_h=[50,10],learning_rate = 0.0,beta1=0.9,beta2=0.999,epsilon=1e-08,opt='Adam'):
    
        
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.opt = opt
        self.n_input = n_featurs
        self.n_h = n_h
        self.Loss =0
        
        
        self.X=X
        
        self.decoder
        self.encoder
        self.Optimizer
        self.biases_init = self.biases_init()
        self.weight_init =  self.weight_init()
        
    def biases_init(self):
        biases_init={}
        biases_init = {
                'encoder_b1': tf.Variable(tf.random_normal([self.n_h[0]])),
                'encoder_b2': tf.Variable(tf.random_normal([self.n_h[1]])),
                'decoder_b1': tf.Variable(tf.random_normal([self.n_h[0]])),
                'decoder_b2': tf.Variable(tf.random_normal([self.n_input]))
                }
        return biases_init
    def weight_init(self):
        weight_init ={}
        weight_init = {
                'encoder_h1': tf.Variable(tf.random_normal([self.n_input,self.n_h[0]] )),
                'encoder_h2': tf.Variable(tf.random_normal([self.n_h[0],self.n_h[1]])),
                'decoder_h1': tf.Variable(tf.random_normal([self.n_h[1],self.n_h[0]])),
                'decoder_h2': tf.Variable(tf.random_normal([self.n_h[0],self.n_input]))
                  }
        return weight_init
        
        
        
    def encoder(self):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.X, self.weight_init['encoder_h1']),
                                       self.biases_init['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weight_init['encoder_h2']),
                                       self.biases_init['encoder_b2']))
        self.encoder = layer_2
        return layer_2
    
    def decoder(self):
        layer_1d = tf.nn.sigmoid(tf.add(tf.matmul(self.encoder(), self.weight_init['decoder_h1']),
                                       self.biases_init['decoder_b1']))
        layer_2d = tf.nn.sigmoid(tf.add(tf.matmul(layer_1d, self.weight_init['decoder_h2']),
                                       self.biases_init['decoder_b2']))
        self.decoder = layer_2d
        return layer_2d

    def loss(self):
        self.Loss += tf.reduce_mean(tf.pow(tf.subtract(self.X, self.decoder() ),2))
        return self.Loss
    def Optimizer(self):
        if self.opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=self.beta1,beta2=self.beta2,epsilon=self.epsilon,use_locking=False).minimize(self.loss())
        elif self.opt=='RMS':    
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss())
        self.Optimizer = optimizer
        return self.Optimizer
        
    
    