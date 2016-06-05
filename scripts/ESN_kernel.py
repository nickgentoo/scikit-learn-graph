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
from skgraph.feature_extraction.graph.ODDSTVectorizerListFeaturesForDeep import ODDSTVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier as PAC
from skgraph.datasets import load_graph_datasets
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
    #working with Chemical
    g_it=load_graph_datasets.dispatch(dataset)
     
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

    PassiveAggressive = PAC(C=0.01)       
    features=Vectorizer.transform([g_it.graphs[i] for i in xrange(50)]) #Parallel ,njobs
    errors=0    
    for i in xrange(features[0].shape[0]): 
        ex=features[0][i]
        if i!=0:
            #W_old contains the model at the preceeding step
            # Here we want so use the deep network to predict the W values of the features 
            # present in ex
            #W=ESN(predict_weights)            
            W=W_old #dump line
            #set the weights of PA to the predicted values
            PassiveAggressive.coef_=W
            pred=PassiveAggressive.predict(ex)
            score=PassiveAggressive.decision_function(ex)

            if pred!=g_it.target[i]:
                errors+=1
                print "Error",errors," on example",i, "pred", score, "target",g_it.target[i]
            else:
                pass
                #print "Correct prediction example",i, "pred", score, "target",g_it.target[i]

        else:
                #first example is always an error!
                errors+=1
                print "Error",errors," on example",i
        #print features[0][i]
        #print features[0][i].shape
        f=features[0][i,:]
        #print f.shape
        #print f.shape
        #print g_it.target[i]    
        #third parameter is compulsory just for the first call
        PassiveAggressive.partial_fit(f,np.array([g_it.target[i]]),np.unique(g_it.target))
        W_old=PassiveAggressive.coef_
    #print GM
#    GMsvm=[]    
#    for i in xrange(len(GM)):
#        GMsvm.append([])
#        GMsvm[i]=[i+1]
#        GMsvm[i].extend(GM[i])
#    #print GMsvm
#    from sklearn import datasets
#    print "Saving Gram matrix"
#    #datasets.dump_svmlight_file(GMsvm,g_it.target, name+".svmlight")
#    datasets.dump_svmlight_file(np.array(GMsvm),g_it.target, name+".svmlight")
#   
