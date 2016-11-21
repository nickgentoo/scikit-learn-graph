# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:02:41 2015

Copyright 2015 Nicolo' Navarin

This file is part of scikit-learn-graph.

scikit-learn-graph is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

scikit-learn-graph is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with scikit-learn-graph.  If not, see <http://www.gnu.org/licenses/>.
"""
import sys
from skgraph.kernel.ODDSTGraphKernel import ODDSTGraphKernel
from skgraph.kernel.ODDSTOrthogonalizedGraphKernel import ODDSTOrthogonalizedGraphKernel
from skgraph.kernel.WLOrthogonalizedGraphKernel import WLOrthogonalizedGraphKernel

from skgraph.datasets import load_graph_datasets
from sklearn import datasets
from sklearn import preprocessing
import numpy as np

from skgraph.datasets.load_graph_datasets import dispatch
#START MIRKO
import scipy.special as spc
import cvxopt as co

def d_kernel(R, k, norm=True):
    
    m = R.size[0]
    n = R.size[1]
    
    x_choose_k = [0]*(n+1)
    x_choose_k[0] = 0
    for i in range(1, n+1):
        x_choose_k[i] = spc.binom(i,k)
    
    nCk = x_choose_k[n]
    X = R*R.T
    
    K = co.matrix(0.0, (X.size[0], X.size[1]))
    for i in range(m):
        for j in range(i, m):
            n_niCk = x_choose_k[n - int(X[i,i])]
            n_njCk = x_choose_k[n - int(X[j,j])]
            n_ni_nj_nijCk = x_choose_k[n - int(X[i,i]) - int(X[j,j]) + int(X[i,j])]
            K[i,j] = K[j,i] = nCk - n_niCk - n_njCk + n_ni_nj_nijCk
    
    if norm:
        YY = co.matrix([K[i,i] for i in range(K.size[0])])
        YY = co.sqrt(YY)**(-1)
        K = co.mul(K, YY*YY.T)

    return K
#END MIRKO

if __name__=='__main__':
    if len(sys.argv)<1:
        sys.exit("python ODDKernel_example.py dataset r l d filename kernel")
    dataset=sys.argv[1]
    max_radius=int(sys.argv[2])
    la=float(sys.argv[3])
    #hashs=int(sys.argv[3])
    njobs=1
    d=int(sys.argv[4]) #MIRKO
    name=str(sys.argv[5])
    kernel=sys.argv[6]
    
    g_it=dispatch(dataset)

     

    if kernel=="ST":
        ODDkernel=ODDSTGraphKernel(r=max_radius,l=la)
    elif kernel=="STOrthogonalized":
        ODDkernel=ODDSTOrthogonalizedGraphKernel(r=max_radius,l=la)
    elif kernel=="WLO":
        ODDkernel=WLOrthogonalizedGraphKernel(r=max_radius)
    else:
        print "warning! uNRECOGNIZED KERNEL!"
    
    feat_list=ODDkernel.transform(g_it.graphs) #Parallel ,njobs
    print feat_list[0],type(feat_list[0])
    
    #codice vecchio
    print "Binarizing features"
    binarizer = preprocessing.Binarizer(threshold=0.0)
    
    binfeatures_list=map(binarizer.transform,feat_list)
    #print binfeatures
    
    #cALCULATE DOT PRODUCT BETWEEN FEATURE REPRESENTATION OF EXAMPLES
    #GM=binfeatures.dot(binfeatures.T).todense().tolist() 
    #print GM
    
    # START MIRKO
    GM=np.zeros((len(g_it.graphs),len(g_it.graphs)))
    for i in range(len(binfeatures_list)):
        print "Calculating D-kernel... for matrix",str(i)
        R = co.matrix(binfeatures_list[i].todense())
        K = d_kernel(R, d)
        GM += np.array(K)#.tolist()
    
    
    #fine codice vecchio
    

    #GM_list=ODDkernel.computeKernelMatrixTrain(g_it.graphs) #Parallel ,njobs
    mat=GM
    GMsvm=[]    
    for i in xrange(len(mat)):
        GMsvm.append([])
        GMsvm[i]=[i+1]
        GMsvm[i].extend(GM_list[mat][i])
    print "Saving Gram matrix"
    datasets.dump_svmlight_file(GMsvm,g_it.target, name+".svmlight")
    datasets.dump_svmlight_file(GMsvm,g_it.target, name+".height"+str(mat)+".svmlight")
    ##print GM
