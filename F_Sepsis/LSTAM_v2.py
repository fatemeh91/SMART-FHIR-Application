#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:04:17 2018

@author: iam
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 11:17:59 2018

@author: fatemeh
"""

import tensorflow as tf
import functools 

from tensorflow.contrib import rnn



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
    def __init__(self,Rewards,C_ST,N_ST,AccLoss,Time_stamp,num_units=1,n_classes=1,n_features=159,learning_rate=0.0001,alpha=0.99):
        self.Rewards=Rewards
        self.C_ST=C_ST
        self.N_ST=N_ST
        self.AccLoss=AccLoss
        self.num_units=num_units
        self.n_classes=n_classes
        self.n_features=n_features
        self.TS=Time_stamp
        self.Acc_Loss=AccLoss
        self.alpha=0.99
        self.learning_rate=learning_rate
        
        
        self.wighto_init
        self.biaso_init
        self.cell
        self.TD_Zero
        self.optimization
        
        
        
    @define_scope
    def wighto_init(self):
        out_weights=tf.Variable(tf.random_normal([self.num_units,self.n_classes]))
        return out_weights
    
    @define_scope
    def biaso_init(self):
        out_bias=tf.Variable(tf.random_normal([self.n_classes]))
        return out_bias
        
    @define_scope
    def cell(self):
        initializer = tf.random_uniform_initializer(-1, 1)
        lstmcell = tf.nn.rnn_cell.LSTMCell(self.num_units, self.n_features, initializer=initializer)
        #cell_out = tf.nn.rnn_cell.OutputProjectionWrapper(cell, self._output_size)
        
        return lstmcell
    @define_scope
    def TD_Zero(self):
        outputs_CS, _ = tf.nn.dynamic_rnn(self.cell, self.C_ST,dtype=tf.float32)
        outputs_NS, _ = tf.nn.dynamic_rnn(self.cell, self.N_ST,dtype=tf.float32)
        TD_error=tf.subtract(tf.add(tf.scalar_mul(self.alpha,outputs_NS[0][0]),self.Rewards[0:-1]),outputs_CS[0][0])
        total_error=tf.reduce_sum(tf.multiply(tf.pow(TD_error,2),(0.5/self.TS)))
        total_error+=tf.multiply(tf.pow(tf.subtract(outputs_NS[0][0][-1],self.Rewards[-1]),2),(0.5/self.TS))
        total_error+=self.Acc_Loss
        return total_error
    @define_scope
    def optimization(self):
        opt=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.TD_Zero/10)
        return opt    
        
        
    
    
        
    
        