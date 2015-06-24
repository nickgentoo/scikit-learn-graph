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
    
#def gaussianKernel(x,z,sigma):
#    return np.exp((-(np.linalg.norm(np.matrix(x)-np.matrix(z))**2))/(2*sigma**2))