#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 17:27:48 2018

@author: iam
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:28:36 2018

@author: iam
"""

# for optmizing a model

from ValueItr_LSTAM_v4_Bays import Main_Loop
import numpy as np
from subprocess import call
import pandas as pd
import os
import sys

def main():

    filepath = 'gs://emory-sepsis-data2/Result_600trainingpatients_500epoch_sampling_Bays_.z'
    filename = './Result_600trainingpatient_500epoch_sampling_Bays.z'
    from functools import partial
    #from skopt.plots import plot_evaluations , plot_objective
    from skopt import gp_minimize, forest_minimize, dummy_minimize,dump,load
    bounds = [(0., 1.),] * 8
    n_calls = 200

    forest_res = gp_minimize(Main_Loop, bounds)
    dump(forest_res,filename)
    
    #data = {'time':'allane','hr':[2.5]}
    #df1 = pd.DataFrame(data)
    #pd.DataFrame.to_csv(df1,os.path.join('./',filename),header=False)
        
    
    call(["gsutil","-m","cp",filename,filepath])
    
    
    
    
    # t=load('./Result.z')
    # _ = plot_evaluations(t)
    # _ = plot_objective(t)


if __name__ == '__main__':
    main()















