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

def plot(data):
    print len(data)
    for x_ind in range(len(data)):
        plt.subplot(10,10,int(x_ind)+1) # column,row, sub 
        plt.plot(data[x_ind], 'o-')
        print x_ind
    
    #plt.subplots_adjust(top=0.92, bottom=0.18, left=0.35, right=0.35, hspace=0.95,
     #               wspace=0.65)
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('function_Value2.png',dpi=100)
    return None


if __name__ == '__main__':
    main()


