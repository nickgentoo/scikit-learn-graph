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
from scipy.sparse import csc_matrix
from sklearn.utils import compute_class_weight
if __name__=='__main__':
    if len(sys.argv)<1:
        sys.exit("python ODDKernel_example.py dataset r l filename kernel C nhidden learningRate")
    dataset=sys.argv[1]
    max_radius=int(sys.argv[2])
    la=float(sys.argv[3])
    #hashs=int(sys.argv[3])
    njobs=1
    name=str(sys.argv[4])
    kernel=sys.argv[5]
    parameterC=float(sys.argv[6])
    nHidden=int(sys.argv[7])
    lr=float(sys.argv[8])
    #FIXED PARAMETERS
    normalization=True
    #working with Chemical
    g_it=load_graph_datasets.dispatch(dataset)
    
    
    f=open(name,'w')
        #generate one-hot encoding
    Features=g_it.label_dict
    tot = len(Features)+3
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
    PassiveAggressive = PAC(C=parameterC,n_iter=1,fit_intercept=False,shuffle=False)       
    features,list_for_deep=Vectorizer.transform(g_it.graphs) #Parallel ,njobs
    errors=0    
    tp=0
    fp=0
    tn=0
    fn=0
    predictions=[0]*50
    correct=[0]*50
    
    
    #print ESN
    model=ESN.EchoStateNetwork(tot,nHidden,1)

    #netDataSet=[]
    #netTargetSet=[]
    #netKeyList=[]
    BERtotal=[]
    bintargets=[1,-1]
    #print features
    #print list_for_deep.keys()

    sep=np.zeros((tot))
    sep[tot-3]=1
    for i in xrange(features.shape[0]):
    	netDataSet=[]
    	netTargetSet=[]
    	netKeyList=[] 
        #i-th example
        #------------ESN dataset--------------------#
        ex=features[i]
        #CREARE W
	for key,rowDict in list_for_deep[i].iteritems():
	  #print "key", key, "target", target
	  
	  netDataSet.append(np.asarray(rowDict))
	  netKeyList.append(key)
	  
        #print "ex", ex
        #print list_for_deep[i].keys()
        
       
        if i!=0:
            #W_old contains the model at the preceeding step
            # Here we want so use the deep network to predict the W values of the features 
            # present in ex
   
            RowIndex=[]
            ColIndex=[]
            matData=[]
           

	    for key,ft in zip(netKeyList,netDataSet):
	      #da una features ottengo una lista di dataset
	 

	      W_key=model.computeOut([ft])[0]#essendo che si passa sempre una solo lemento in una lista avro sempre un solo output

	      RowIndex.append(0)#W deve essere è un vettore riga
	      ColIndex.append(key)
	      matData.append(W_key)
	  
		
		
	    #print "N_features", ex.shape
	    W=np.asarray(csc_matrix((matData, (RowIndex, ColIndex)), shape=ex.shape).todense())
	    #print W
	    #raw_input("W (output)")
	

	    #W=W_old #dump line

            
            #set the weights of PA to the predicted values
            PassiveAggressive.coef_=W
            pred=PassiveAggressive.predict(ex)
 
            score=PassiveAggressive.decision_function(ex)

	    bintargets.append(g_it.target[i])
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
		pred=0
		score=0
                errors+=1
                print "Error",errors," on example",i
                if g_it.target[i]==1:
                    fn+=1
                else:
                    fp+=1
	#print i
        if i%50==0 and i!=0:
                #output performance statistics every 50 examples
                if (tn+fp) > 0:
                    pos_part= float(fp) / (tn+fp)
                else:
                    pos_part=0
                if (tp+fn) > 0:
                    neg_part=float(fn) / (tp+fn)
                else:
                    neg_part=0
                BER = 0.5 * ( pos_part  + neg_part)    
                print "1-BER Window esempio ",i, (1.0 - BER)
                print>>f,"1-BER Window esempio "+str(i)+" "+str(1.0 - BER)
                BERtotal.append(1.0 - BER)
                tp = 0
                fp = 0
                fn = 0
                tn = 0
		bintargets=[1,-1]
        #print features[0][i]
        #print features[0][i].shape
        #f=features[0][i,:]
        #print f.shape
        #print f.shape
        #print g_it.target[i]    
        #third parameter is compulsory just for the first call
	print "prediction", pred, score
	#print "intecept",PassiveAggressive.intercept_
	#raw_input()
        if abs(score)<1.0 or pred!=g_it.target[i]:
	
	    ClassWeight=compute_class_weight('auto',np.asarray([1,-1]),bintargets)
	    #print "class weights", {1:ClassWeight[0],-1:ClassWeight[1]}
	    PassiveAggressive.class_weight={1:ClassWeight[0],-1:ClassWeight[1]}

	    PassiveAggressive.partial_fit(ex,np.array([g_it.target[i]]),np.unique(g_it.target))
	    #PassiveAggressive.partial_fit(ex,np.array([g_it.target[i]]),np.unique(g_it.target))
	    W_old=PassiveAggressive.coef_
	    
	    
	    #ESN target---#
	    netTargetSet=[]
	    for key,rowDict in list_for_deep[i].iteritems():


		target=np.asarray( [np.asarray([W_old[0,key]])]*len(rowDict))
		
	      
		netTargetSet.append(target)


	    

	    #------------ESN TargetSetset--------------------#
	    # ESN Training
	    
	    #for ftDataset,ftTargetSet in zip(netDataSet,netTargetSet):
	    #print "Input"
	    #print netDataSet
	    #raw_input("Output")
	    #print netTargetSet
	    #raw_input("Target")
	    model.OnlineTrain(netDataSet,netTargetSet,lr)
	    #raw_input("TR")
	    #calcolo statistiche

print "BER AVG", sum(BERtotal) / float(len(BERtotal))
print>>f,"BER AVG "+str(sum(BERtotal) / float(len(BERtotal)))
f.close()
