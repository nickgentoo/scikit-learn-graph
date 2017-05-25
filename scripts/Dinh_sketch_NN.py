# -*- coding: utf-8 -*-
"""


python -m scripts/Online_PassiveAggressive_countmeansketch LMdata 3 1 a ODDST 0.01  

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
from copy import copy
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import sys
from skgraph.feature_extraction.graph.ODDSTVectorizer import ODDSTVectorizer
from skgraph.feature_extraction.graph.WLVectorizer import WLVectorizer

from skgraph.datasets import load_graph_datasets
import numpy as np
from scipy.sparse import csc_matrix
from sklearn.utils import compute_class_weight
from scipy.sparse import csr_matrix
from skgraph.utils.hasherforNN import NNhasher
from itertools import izip
import time
if __name__=='__main__':
    start_time = time.time()

    if len(sys.argv)<1:
        sys.exit("python ODDKernel_example.py dataset r l out_filename kernel m")
    dataset=sys.argv[1]
    max_radius=int(sys.argv[2])
    la=float(sys.argv[3])
    #hashs=int(sys.argv[3])
    njobs=1
    name=str(sys.argv[4])
    kernel=sys.argv[5]
    m=int(sys.argv[6])

    #lr=float(sys.argv[7])
    #FIXED PARAMETERS
    normalization=False
    #working with Chemical
    g_it=load_graph_datasets.dispatch(dataset)
    
    
    f=open(name,'w')
    

    
    #At this point, one_hot_encoding contains the encoding for each symbol in the alphabet
    if kernel=="WL":
        print "Lambda ignored"
        print "Using WL fast subtree kernel"
        Vectorizer=WLVectorizer(r=max_radius,normalization=normalization)
    elif kernel=="ODDST":
        print "Using ST kernel"
        Vectorizer=ODDSTVectorizer(r=max_radius,l=la,normalization=normalization)
    elif kernel=="NSPDK":
        print "Using NSPDK kernel, lambda parameter interpreted as d"
        Vectorizer=NSPDKVectorizer(r=max_radius,d=int(la),normalization=normalization)
    else:
        print "Unrecognized kernel"
    #TODO the C parameter should probably be optimized


    #print zip(_letters, _one_hot)
    #exit()
    features=Vectorizer.transform(g_it.graphs) #Parallel ,njobs
    print "examples, features", features.shape
    features_time=time.time()
    print("Computed features in %s seconds ---" % (features_time - start_time))

    #generate the weight matrix, it also has a parameter rs that is the random seed. leave this like it is for now
    hasher=NNhasher(features.shape[1], m)
    #you can get the weight matrix with the following method
    weight_matrix=hasher.getMatrix()
    # this for loop is just for showing you the functions. Probably you want to split the dataset in training and test.
    for i in xrange(features.shape[0]):
          time1=time.time()

          ex=features[i][0]
          #ex is a scipy.sparse.csr.csr_matrix of shape (1, number_of_features)
          #print type(ex), ex.shape
          #generate the hidden representation of the example, you should not use this function for now
          #exHidden=hasher.transform(ex)
          #print exHidden
          #print "exCMS", type(exCMS), exCMS.shape
          target=g_it.target[i]

          #----------------------------
          #ex is your example, target is its target. Here you have to insert the actual code to perform learning.




end_time=time.time()
print("Learning phase time %s seconds ---" % (end_time - features_time )) #- cms_creation
print("Total time %s seconds ---" % (end_time - start_time))
