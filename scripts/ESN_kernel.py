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
sys.path.insert(0, 'ESN/Net')
import ESN_2p0_4Kernel as ESN
from skgraph.feature_extraction.graph.ODDSTVectorizerListFeaturesForDeep import ODDSTVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier as PAC
from skgraph.datasets import load_graph_datasets
import numpy as np

if __name__=='__main__':
    if len(sys.argv)<1:
        sys.exit("python ODDKernel_example.py dataset r l filename kernel nhidden")
    dataset=sys.argv[1]
    max_radius=int(sys.argv[2])
    la=float(sys.argv[3])
    hashs=int(sys.argv[3])
    njobs=1
    name=str(sys.argv[4])
    kernel=sys.argv[5]
    nHidden=int(sys.argv[6])
    #FIXED PARAMETERS
    normalization=True
    #working with Chemical
    g_it=load_graph_datasets.dispatch(dataset)
    
        #generate one-hot encoding
    Features=g_it.label_dict
    tot = len(Features)+2
    print "Total number of labels", tot
    _letters=[]    
    _one_hot=[]
    for key,n in Features.iteritems():
        #print "key",key,"n", n
        #generare numpy array, np.zeros((dim))[n]=1
        a=np.zeros((tot))
        a[n-1]=1
        _one_hot.append(a)
        #_one_hot.append("enc"+str(n))
        #_one_hot.append(' '.join(['0']*(n-1) + ['1'] + ['0']*(tot-n)))
        _letters.append(key)
    a=np.zeros((tot))
    a[tot-2]=1
    _one_hot.append(a)
    #_one_hot.append("enc"+str(tot-1))
    a=np.zeros((tot))
    a[tot-1]=1
    _one_hot.append(a)

    #_one_hot.append("enc"+str(tot))

    #_one_hot.append(' '.join(['0']*(tot-2) + ['1'] + ['0']*(1)))
    #_one_hot.append(' '.join(['0']*(tot-1) + ['1'] + ['0']*(0)))
    _letters.append("P")
    _letters.append("N")
    one_hot_encoding = dict(zip(_letters, _one_hot))
    
    #At this point, one_hot_encoding contains the encoding for each symbol in the alphabet
    if kernel=="WL":
        print "Lambda ignored"
        print "Using WL fast subtree kernel"
        Vectorizer=WLVectorizer(r=max_radius,normalization=normalization)
    elif kernel=="ODDST":
        print "Using ST kernel"
        Vectorizer=ODDSTVectorizer(r=max_radius,l=la,normalization=normalization, one_hot_encoding=one_hot_encoding)
    elif kernel=="NSPDK":
        print "Using NSPDK kernel, lambda parameter interpreted as d"
        Vectorizer=NSPDKVectorizer(r=max_radius,d=int(la),normalization=normalization)
    else:
        print "Unrecognized kernel"
    #TODO the C parameter should probably be optimized


    #print zip(_letters, _one_hot)
    #exit()
    PassiveAggressive = PAC(C=0.01)       
    features,list_for_deep=Vectorizer.transform([g_it.graphs[i] for i in xrange(50)]) #Parallel ,njobs
    errors=0    
    tp=0
    fp=0
    tn=0
    fn=0
    predictions=[0]*50
    correct=[0]*50
    
    

    model=ESN.EchoStateNetwork(tot,nHidden,1)
    netDataSet=[]
    netTargetSet=[]
    for i in xrange(features.shape[0]): 
        #i-th example
        ex=features[i]
        #------------ESN dataset--------------------#
	#print features
        #for key,rowDict in list_for_deep[i].iteritems():
	  #target=features[key]
	  #for festuresList in rowDict:
	      #netDataSet.append(np.array(festuresList)).shape
        #raw_input()
        #------------ESN dataset--------------------#
        if i!=0:
            #W_old contains the model at the preceeding step
            # Here we want so use the deep network to predict the W values of the features 
            # present in ex
            #W=model.computeOut()#ESN(predict_weights)            
            W=W_old #dump line
            #set the weights of PA to the predicted values
            PassiveAggressive.coef_=W
            pred=PassiveAggressive.predict(ex)
            score=PassiveAggressive.decision_function(ex)

            if pred!=g_it.target[i]:
                errors+=1
                print "Error",errors," on example",i, "pred", score, "target",g_it.target[i]
                if g_it.target[i]==1:
                    fn+=1
                else:
                    fp+=1
            
            else:
                if g_it.target[i]==1:
                    tp+=1
                else:
                    tn+=1
                #print "Correct prediction example",i, "pred", score, "target",g_it.target[i]

        else:
                #first example is always an error!
                errors+=1
                print "Error",errors," on example",i
        #print features[0][i]
        #print features[0][i].shape
        #f=features[0][i,:]
        #print f.shape
        #print f.shape
        #print g_it.target[i]    
        #third parameter is compulsory just for the first call
        PassiveAggressive.partial_fit(ex,np.array([g_it.target[i]]),np.unique(g_it.target))
        W_old=PassiveAggressive.coef_
        # PASSO DI APPRENDIMENTO DELLA DEEP        
        
        #calcolo statistiche
           
