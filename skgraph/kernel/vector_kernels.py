# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 17:25:57 2015

@author: Nicolò Navarin
"""
from scipy.spatial.distance import pdist
import numpy as np

def gaussianKernel(X,Y,beta):
    Z=[X,Y]
    return np.exp(-beta * pdist(Z, 'sqeuclidean'))