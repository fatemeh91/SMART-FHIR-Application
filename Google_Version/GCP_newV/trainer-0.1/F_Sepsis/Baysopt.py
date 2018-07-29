#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:28:36 2018

@author: iam
"""

from GPyOpt.methods import BayesianOptimization
from Value_Prediction_Baysopt import Main_Loop
import numpy as np

import numpy as np
from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO



def main():

    def f(x):
        return (np.sin(x))
    
    
    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode='ExpectedImprovement')
    param = {'n_hidden_2': ('int', [80,120]),'aOp'}
    
    np.random.seed(23)
    gpgo = GPGO(gp, acq, Main_Loop, param)
    gpgo.run(max_iter=20)
    res = gpgo.getResult()[0]
    print res

    
#    def f(x):
#        return (6*x-2)**2*np.sin(12*x-4)
#    bounds =[ {'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}]
#    myBopt = BayesianOptimization(f,bounds)
#    myBopt.run_optimization(max_iter=15)

    #bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}]
    #myBopt = BayesianOptimization(f=f, bounds=bounds)
    #myBopt.run_optimization(max_iter=15)


if __name__ == '__main__':
    main()















