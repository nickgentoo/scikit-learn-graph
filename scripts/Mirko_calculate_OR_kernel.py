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
from skgraph.feature_extraction.graph.ODDSTVectorizer import ODDSTVectorizer
from skgraph.feature_extraction.graph.NSPDK.NSPDKVectorizer import NSPDKVectorizer
from skgraph.feature_extraction.graph.WLVectorizer import WLVectorizer
from sklearn import preprocessing

from skgraph.datasets.load_graph_datasets import dispatch
import numpy as np

if __name__=='__main__':
    if len(sys.argv)<1:
        sys.exit("python ODDKernel_example.py dataset r l filename kernel")
    dataset=sys.argv[1]
    max_radius=int(sys.argv[2])
    la=float(sys.argv[3])
    #hashs=int(sys.argv[3])
    njobs=1
    name=str(sys.argv[4])
    kernel=sys.argv[5]
    #FIXED PARAMETERS
    normalization=True
    g_it=dispatch(dataset)
    

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
       
    features=Vectorizer.transform(g_it.graphs) #Parallel ,njobs
    
    #INSERT CODE HERE TO MODIFY FEATURES
    # features is a n_examplesxn_features sparse matrix
    #binarize features
    print "Binarizing features"
    binarizer = preprocessing.Binarizer(threshold=0.0)
    
    binfeatures=binarizer.transform(features)
    print binfeatures
    #cALCULATE DOT PRODUCT BETWEEN FEATURE REPRESENTATION OF EXAMPLES
    GM=binfeatures.dot(binfeatures.T).todense().tolist() 
    #print GM
    
    print "Saving Gram matrix"
    output=open(name+".svmlight","w")
    for i in xrange(len(GM)):
        output.write(str(g_it.target[i])+" 0:"+str(i+1)+" ")
        for j in range(len(GM[i])):
            output.write(str(j+1)+":"+str(GM[i][j])+" ")
        output.write("\n")
    output.close()
    #print GMsvm
    from sklearn import datasets
#    #print GM
##    GMsvm=[]    
##    for i in xrange(len(GM)):
##        GMsvm.append([])
##        GMsvm[i]=[i+1]
##        GMsvm[i].extend(GM[i])
##    #print GMsvm
##    from sklearn import datasets
##    print "Saving Gram matrix"
##    #datasets.dump_svmlight_file(GMsvm,g_it.target, name+".svmlight")
##    datasets.dump_svmlight_file(np.array(GMsvm),g_it.target, name+".svmlight")
##    #Test manual dump
#    print "Extracted", features.shape[1], "features from",features.shape[0],"examples."
#    print "Saving Features in svmlight format in", name+".svmlight"
#    #print GMsvm
#    from sklearn import datasets
#    datasets.dump_svmlight_file(features,g_it.target, name+".svmlight", zero_based=False)
#    #print GM