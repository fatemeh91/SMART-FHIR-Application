#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:43:06 2018

@author: iam
"""
import numpy as np
def FloatOrNone(value):
    
    try:
        return float(value)
    except:
        return float(0.0)    