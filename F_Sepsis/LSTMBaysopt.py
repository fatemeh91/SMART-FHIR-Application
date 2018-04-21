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

from GPyOpt.methods import BayesianOptimization
from ValueItr_LSTAM_V2_Bays import Main_Loop
import numpy as np

import numpy as np
from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO



def main():

    
    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode='ExpectedImprovement')
    param = {'r1':  ('cont', (0, 1)),'r2':  ('cont', (0, 1)),'r3':  ('cont', (0, 1)),'r4':  ('cont', (0, 1)),'r5':  ('cont', (0, 1)),'r6':  ('cont', (0, 1)),'r7':  ('cont', (0, 1)),'r8':  ('cont', (0, 1))}
    
    gpgo = GPGO(gp, acq, Main_Loop, param)
    gpgo.run(max_iter=200)
    res = gpgo.getResult()[0]
    print res



if __name__ == '__main__':
    main()















