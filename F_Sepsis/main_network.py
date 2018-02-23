#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:56:42 2018

@author: fatemeh
"""
import tensorflow as tf
class main_network:

    def __init__(self, n_featurs,n_h=[100,200],n_layers=2,n_class=1):
        self.n_input=n_featurs
        self.n_h = n_h
        self.n_layers = n_layers 
        self.n_class = n_class 
    
        self.X = tf.placeholder("float", [None, n_featurs]) # fo each patients at each time step
        self.Y = tf.placeholder("float", [None, 1]) # for each state of each patients at each time step

    def construct_back_bone_nn(self):
        weights = []
        biase = []
        # constructing first layer
        l1_weights = tf.Variable(tf.random_normal([self.n_input, self.n_h[0]])) # fully connected neural networks 
        l1_bias=tf.Variable(tf.random_normal([self.n_h[0]]))
        
        weights.append(l1_weights)
        biase.append(l1_bias)
        
        layer_no =0 
        while (layers_no < self.n_layers-1) :
            l_weights = tf.Variable(tf.random_normal([self.n_h[layers_no], self.n_h[layers_no+1]]))
            l_bias=tf.Variable(tf.random_normal([n_h[layers_no+1]]))
            weights.append(l_weights)
            biase.append(l_bias)
            layers_no +=1
        
        
        ## constructing out putlayer
            l_weights = tf.Variable(tf.random_normal([self.n_h[layers_no], self.n_class]))
            l_bias=tf.Variable(tf.random_normal([self.n_class]))
            biase.append(l_bias)
            weights.append(l_weights)
        
        return self
                    
         
         # simple neural  network with out aouto encoder
    def feed_forward_neural_net(self):
         out_layer = tf.nn.sigmoid(tf.add(tf.matmul(self.X, weights[0]),biases[0]))
         layers_no =1 
         while layer_no< self.n_layers :
            out_layer = tf.nn.sigmoid(tf.add(tf.matmul(out_layer,weights[layer_no]),biases[layer_no]))
            layers_no +=1 
            
        
         return out_layer
    