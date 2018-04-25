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

# this version of LSTAM  this version has for problem 

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
    def __init__(self,Rewards,C_ST,AccLoss,n_L_B=1,n_features=159,learning_rate=0.0001):
        self.C_ST=C_ST
        self.R=Rewards
        self.n_step=n_L_B
        self.n_features=n_features
        self.AccLoss=AccLoss
        self.alpha=0.99
        self.learning_rate = learning_rate
        
        #forget unit
        self.weightf_init
        self.biasesf_init
        
        #input unit
        self.weighti_init
        self.biasesi_init
        
        #control unit
        self.weightz_init
        self.biasesz_init
        
        
        
        #peephole wights
        self.peepholei_init
        self.peepholef_init
        self.peepholeo_init
        
        #recurents weights 
        self.reccurenti_init
        self.reccurentz_init
        self.reccurentf_init
        self.reccurento_init
        
        self.TD_Zero 
        self.optimization
        self.prediction
        
    #input wight     
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def weightf_init(self):
        
        weights = tf.Variable(tf.truncated_normal([self.n_step, self.n_features],stddev=0.001)) # fully connected neural networks 
        return weights
    
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def weighto_init(self):
        
        weights = tf.Variable(tf.truncated_normal([self.n_step, self.n_features],stddev=0.001)) # fully connected neural networks 
        return weights
    
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def weighti_init(self):
        
        weights = tf.Variable(tf.truncated_normal([self.n_step, self.n_features],stddev=0.001)) # fully connected neural networks 
        return weights
    
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def weightz_init(self):
        
        weights = tf.Variable(tf.truncated_normal([self.n_step, self.n_features],stddev=0.001)) # fully connected neural networks 
        return weights


    # recuurent weight 
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def reccurenti_init(self):
        
        weights = tf.Variable(tf.truncated_normal([self.n_step, self.n_step],stddev=0.001)) # fully connected neural networks 
        return weights

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def reccurentz_init(self):
        
        weights = tf.Variable(tf.truncated_normal([self.n_step, self.n_step],stddev=0.001)) # fully connected neural networks 
        return weights
    
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def reccurentf_init(self):
        
        weights = tf.Variable(tf.truncated_normal([self.n_step, self.n_step],stddev=0.001)) # fully connected neural networks 
        return weights

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def reccurento_init(self):
        
        weights = tf.Variable(tf.truncated_normal([self.n_step, self.n_step],stddev=0.001)) # fully connected neural networks 
        return weights
    

    #biases wight    
    @define_scope
    def biasesf_init(self):   
        biase=tf.Variable(tf.zeros(self.n_step))
        return biase

    
    @define_scope
    def biaseso_init(self):   
        biase=tf.Variable(tf.zeros(self.n_step))
        return biase
    
    @define_scope
    def biasesi_init(self):   
        biase=tf.Variable(tf.zeros(self.n_step))
        return biase
    
    @define_scope
    def biasesz_init(self):   
        biase=tf.Variable(tf.zeros(self.n_step))
        return biase    
    
    
    
    #peephole wight
    @define_scope
    def peepholef_init(self):   
        weight=tf.Variable(tf.truncated_normal([self.n_step],stddev=0.001))
        return weight

    @define_scope
    def peepholeo_init(self):   
        weight=tf.Variable(tf.truncated_normal([self.n_step],stddev=0.001))
        return weight

    @define_scope
    def peepholei_init(self):   
        weight=tf.Variable(tf.truncated_normal([self.n_step],stddev=0.001))
        return weight
    
    @define_scope
    def TD_Zero(self):
        
        # block input 
        def block_input(X,Ym1):
            Z_hat=tf.matmul(Ym1,self.reccurentz_init)#tf.add(tf.add(tf.matmul(self.weightz_init,X),tf.matmul(Ym1,self.reccurentz_init)),self.biasesz_init)
            Z=tf.tanh(Z_hat)
            return Z
        # input gate
        def input_gate(X,Ym1,cellm1):
            I_hat=tf.add(tf.dd(tf.add(tf.matmul(self.weighti_init,self.X),tf.matmul(self.reccurenti_init,Ym1)),tf.matmul(self.peepholei_init,cellm1)),self.biasesi_init)
            I=tf.sigmoid(I_hat)
            return I
        
        # forget gate
        def forget_gate(X,Ym1,cellm1):
            F_hat=tf.add(tf.dd(tf.add(tf.matmul(self.weightf_init,X),tf.matmul(self.reccurentf_init,Ym1)),tf.matmul(self.peepholef_init,cellm1)),self.biasesf_init)
            F=tf.sigmoid(F_hat)
            return F
        # cell
        def cell(X,Ym1,cellm1):
            cell=tf.add(tf.matmul(block_input(X,Ym1),input_gate(X,Ym1,cellm1)),tf.matmul(cellm1,forget_gate(X,Ym1,cellm1)))
            return cell
        # output gate
        def output_gate(X,Ym1,cellm1):
            O_hat=tf.add(tf.dd(tf.add(tf.matmul(self.weighto_init,X),tf.matmul(self.reccurento_init,Ym1)),tf.matmul(self.peepholeo_init,cell(X,Ym1,cellm1))),self.biaseso_init)
            O=tf.sigmoid(O_hat)
            return O
        # bolck output
        def block_output(X,Ym1,cellm1):
            y=tf.matmul(tf.tanh(cell(X,Ym1,cellm1)),output_gate(X,Ym1,cellm1))
            return y
        
        #unstack_C_ST=tf.unstack(self.C_ST,0)
        unstack_C_ST= self.C_ST #tf.split(self.C_ST,self.N_TS)
        C_V=tf.cast(0, tf.float32)
        cellm1=tf.cast(0, tf.float32)
        TD_error=tf.cast(0, tf.float32)
        
        for X_ind in range(0,tf.shape(self.C_ST)[0]-1):
            cell_V=cell(unstack_C_ST[X_ind],C_V,cellm1)
            C_V=block_output(unstack_C_ST[X_ind],C_V,cellm1)
            cellm1=cell_V
            cell_V=cell(unstack_C_ST[X_ind],C_V,cellm1)
            N_V=block_output(unstack_C_ST[X_ind+1],C_V,cellm1)
            TD_error+=tf.subtract(tf.add(tf.scalar_mul(self.alpha,N_V),self.R[X_ind]),C_V)
            
        
        #N_V=block_output(unstack_C_ST[len(unstack_C_ST)-1],C_V,cellm1)
        #TD_error+=tf.subtract(N_V,self.R[-1])
        TD_error+=self.AccLoss
            
#        Val_C_ST=tf.convert_to_tensor(C_V[1:])
#        
#        N_V=[]
#        N_V.append(0)
#        cellm1=[]
#        cellm1.append(0)
#        for X in unstack_N_ST:
#            cellm1.append(cell(X,N_V[-1],cellm1[-1]))
#            N_V.append(block_output(X,N_V[-1],cellm1[-2]))
#        Val_N_ST=N_V[1:]
#        
#        
#        TD_error = tf.subtract(tf.add(tf.scalar_mul(self.alpha,Val_C_ST),self.R[0:-1]),Val_N_ST)
#        TD_error+=self.AccLoss
        
        return TD_error
    
    @define_scope
    def optimization(self):
        opt=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.TD_Zero/10)
        return opt   
                
    @define_scope
    def prediction(self):
        # block input 
        def block_input(X,Ym1):
            Z_hat=tf.dd(tf.add(tf.mat_mul(self.weightz_init,X),tf.mat_mul(self.reccurentz_init,Ym1)),self.biasesz_init)
            Z=tf.tanh(Z_hat)
            return Z
        # input gate
        def input_gate(X,Ym1,cellm1):
            I_hat=tf.add(tf.dd(tf.add(tf.mat_mul(self.weighti_init,self.X),tf.mat_mul(self.reccurenti_init,Ym1)),tf.mat_mul(self.peepholei_init,cellm1)),self.biasesi_init)
            I=tf.sigmoid(I_hat)
            return I
        
        # forget gate
        def forget_gate(X,Ym1,cellm1):
            F_hat=tf.add(tf.dd(tf.add(tf.mat_mul(self.weightf_init,X),tf.mat_mul(self.reccurentf_init,Ym1)),tf.mat_mul(self.peepholef_init,cellm1)),self.biasesf_init)
            F=tf.sigmoid(F_hat)
            return F
        # cell
        def cell(X,Ym1,cellm1):
            cell=tf.add(tf.matmul(block_input(X,Ym1),input_gate(X,Ym1,cellm1)),tf.matmul(cellm1,forget_gate(X,Ym1,cellm1)))
            return cell
        # output gate
        def output_gate(X,Ym1,cellm1):
            O_hat=tf.add(tf.dd(tf.add(tf.mat_mul(self.weighto_init,X),tf.mat_mul(self.reccurento_init,Ym1)),tf.mat_mul(self.peepholeo_init,cell(X,Ym1,cellm1))),self.biaseso_init)
            O=tf.sigmoid(O_hat)
            return O
        # bolck output
        def block_output(X,Ym1,cellm1):
            y=tf.matmul(tf.tanh(cell(X,Ym1,cellm1)),output_gate(X,Ym1,cellm1))
            return y
        
        unstack_C_ST=tf.unstack(self.C_ST,0)
        C_V=[]
        C_V.append(0)
        cellm1=[]
        cellm1.append(0)
        for X in unstack_C_ST:
            cellm1.append(cell(X,C_V[-1],cellm1[-1]))
            C_V.append(block_output(X,C_V[-1],cellm1[-2]))
            
        Val_C_ST=C_V[1:]
        return Val_C_ST
        
        
        
    
    
    
    
    
    
    
    
        
    
        