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





class Auto_Encoder(object):
    
    def __init__(self,X,n_featurs=35,n_h=[128,70],learning_rate = 0.0,beta1=0.9,beta2=0.999,epsilon=1e-08,opt='RMS'):
        self.graph=tf.Graph()
        self.X=X
        with self.graph.as_default():
            self.n_h = n_h
            self.n_input = n_featurs
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.opt = opt
            #operations
            self.weight_init =  {
                    'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_h[0]] )),
                    'encoder_h2': tf.Variable(tf.random_normal([self.n_h[0],self.n_h[1]])),
                    'decoder_h1': tf.Variable(tf.random_normal([self.n_h[1], self.n_h[0]])),
                    'decoder_h2': tf.Variable(tf.random_normal([self.n_h[0],self.n_input])),
                    } 
            
            self.biases_init = {
                'encoder_b1': tf.Variable(tf.random_normal([self.n_h[0]])),
                'encoder_b2': tf.Variable(tf.random_normal([self.n_h[1]])),
                'decoder_b1': tf.Variable(tf.random_normal([self.n_h[0]])),
                'decoder_b2': tf.Variable(tf.random_normal([self.n_input])),
                }
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.X, self.weight_init['encoder_h1']),
                                       self.biases_init['encoder_b1']))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weight_init['encoder_h2']),
                                       self.biases_init['encoder_b2']))
            
            self.encoder  = layer_2
            
            layer_1d = tf.nn.sigmoid(tf.add(tf.matmul(self.encoder, self.weight_init['decoder_h1']),
                                       self.biases_init['decoder_b1']))
            # Decoder Hidden layer with sigmoid activation #2
            layer_2d = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weight_init['decoder_h2']),
                                       self.biases_init['decoder_b2']))
        
        
        
            self.decoder = layer_2d 
            self.Loss = tf.reduce_mean(tf.pow(tf.subtract(self.X, self.decoder), 2))
            self.Accumulative_loss =tf.cond(self.Accumulatve_loss==None,tf.add(tf.reduce_mean(tf.pow(tf.subtract(self.X, self.decoder), 2)),0)
                                            ,tf.add(tf.reduce_mean(tf.pow(tf.subtract(self.X, self.decoder), 2)),
                                                                           self.Accumulative_loss))
            
            if self.opt == 'Adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=self.beta1,beta2=self.beta2,epsilon=self.epsilon,use_locking=False).minimize(self.Accumulative_loss)
            elif self.opt=='RMS':    
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.Accumulative_loss)
            self.Optimizer =optimizer
            
        #to lunch the graph
        def launchG(self, AE_DATA_X,batch_size):
            with tf.Session(graph=self.graph) as sess:
                sess.run(tf.initialize_all_variables())
                ACC_Loss = [0]
                for batch_ind in range(int(len(AE_DATA_X))/batch_size):
                    for x_ind in range( batch_ind*batch_size , batch_ind*batch_size+batch_size):
                        ACC_Loss =[sess.run(self.Accumulative_Loss,feed_dict={X:AE_DATA_X[x_ind],Feature_no:len(AE_DATA_X[0][0])})[0]]
                        print ACC_Loss
                    print(sess.run(self.Optimizer,feed_dict={self.X:AE_DATA_X[x_ind+1],self.Feature_no:len(AE_DATA_X[0][0])}))
                