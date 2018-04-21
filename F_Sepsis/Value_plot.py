#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 12:07:04 2018

@author: iam
"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    plot()
    return None

def plot(X):
    
    for x_ind in range(len(X)):
        print str(210+int(x_ind)+1)
        plt.subplot(20,5,int(x_ind)+1) # column,row, sub 
        plt.plot(X[x_ind], 'o-')
        plt.show()
        plt.savefig('function_Value.png')
    return None


if __name__ == '__main__':
    main()


