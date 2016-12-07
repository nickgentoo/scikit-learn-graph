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
import sys
from skgraph.feature_extraction.graph.ODDSTVectorizer import ODDSTVectorizer
from skgraph.feature_extraction.graph.WLVectorizer import WLVectorizer

from sklearn.linear_model import PassiveAggressiveClassifier as PAC
from skgraph.datasets import load_graph_datasets
import numpy as np
from scipy.sparse import csc_matrix
from sklearn.utils import compute_class_weight
from scipy.sparse import csr_matrix
from countminsketch import CountMinSketch

if __name__=='__main__':
    if len(sys.argv)<1:
        sys.exit("python ODDKernel_example.py dataset r l filename kernel C m d")
    dataset=sys.argv[1]
    max_radius=int(sys.argv[2])
    la=float(sys.argv[3])
    #hashs=int(sys.argv[3])
    njobs=1
    name=str(sys.argv[4])
    kernel=sys.argv[5]
    C=float(sys.argv[6])
    m=int(sys.argv[7])
    d=int(sys.argv[7])

    #lr=float(sys.argv[7])
    #FIXED PARAMETERS
    normalization=True
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
    
    errors=0    
    tp=0
    fp=0
    tn=0
    fn=0
    predictions=[0]*50
    correct=[0]*50
    
    
    #print ESN

    #netDataSet=[]
    #netTargetSet=[]
    #netKeyList=[]
    BERtotal=[]
    bintargets=[1,-1]
    #print features
    #print list_for_deep.keys()
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    part_plus=0
    part_minus=0
    WCMS=CountMinSketch(m,d)
    for i in xrange(features.shape[0]):

          ex=features[i][0]
          W=csr_matrix(ex)

          rows,cols = ex.nonzero()
          dot=0.0
          for row,col in zip(rows,cols):
              ((row,col), ex[row,col])
              #print col, ex[row,col]
              dot+=WCMS[col]*ex[row,col]
              #print dot
              #TODO aggiungere bias
          if (np.sign(dot) != g_it.target[i] ):
             #print "error on example",i, "predicted:", dot, "correct:", g_it.target[i]
             errors+=1
             if g_it.target[i]==1:
                     fn+=1
             else:
                     fp+=1
          else:
             #print "correct classification", g_it.target[i]
             if g_it.target[i]==1:
                    tp+=1
             else:
                     tn+=1
          if(g_it.target[i]==1):
              coef=(part_minus+1.0)/(part_plus+part_minus+1.0)
              part_plus+=1
          else:
              coef=(part_plus+1.0)/(part_plus+part_minus+1.0)
              part_minus+=1
          tao = min (C, max (0.0,(1.0 - g_it.target[i]*dot )*coef) );
          
          if (tao > 0.0):
              for row,col in zip(rows,cols):
                   ((row,col), ex[row,col])
                   #print col, ex[row,col]
                   WCMS.add(col,g_it.target[i]*tao*ex[row,col])

                 #print "Correct prediction example",i, "pred", score, "target",g_it.target[i]
 

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
                #print>>f,"1-BER Window esempio "+str(i)+" "+str(1.0 - BER)
                BERtotal.append(1.0 - BER)
                tp = 0
                fp = 0
                fn = 0
                tn = 0
                part_plus=0
                part_minus=0
print "BER AVG", np.average(BERtotal)
         
          #print "N_features", ex.shape
        #generate explicit W from CountMeanSketch 
         #print W
        #raw_input("W (output)")
#==============================================================================
#     
#           tao = /*(double)labels->get_label(idx_a) **/ min (C, max (0.0,(1.0 - (((double)labels->get_label(idx_a))*(classe_mod) )) * c_plus ) / modulo_test);
# 
#         #W=W_old #dump line
# 
#             
#             #set the weights of PA to the predicted values
#             PassiveAggressive.coef_=W
#             pred=PassiveAggressive.predict(ex)
#  
#             score=PassiveAggressive.decision_function(ex)
# 
#         bintargets.append(g_it.target[i])
#             if pred!=g_it.target[i]:
#                 errors+=1
#                 print "Error",errors," on example",i, "pred", score, "target",g_it.target[i]
#                 if g_it.target[i]==1:
#                     fn+=1
#                 else:
#                     fp+=1
#             
#             else:
#                 if g_it.target[i]==1:
#                     tp+=1
#                 else:
#                     tn+=1
#                 #print "Correct prediction example",i, "pred", score, "target",g_it.target[i]
# 
#         else:
#                 #first example is always an error!
#         pred=0
#         score=0
#                 errors+=1
#                 print "Error",errors," on example",i
#                 if g_it.target[i]==1:
#                     fn+=1
#                 else:
#                     fp+=1
#     #print i
#         if i%50==0 and i!=0:
#                 #output performance statistics every 50 examples
#                 if (tn+fp) > 0:
#                     pos_part= float(fp) / (tn+fp)
#                 else:
#                     pos_part=0
#                 if (tp+fn) > 0:
#                     neg_part=float(fn) / (tp+fn)
#                 else:
#                     neg_part=0
#                 BER = 0.5 * ( pos_part  + neg_part)    
#                 print "1-BER Window esempio ",i, (1.0 - BER)
#                 print>>f,"1-BER Window esempio "+str(i)+" "+str(1.0 - BER)
#                 BERtotal.append(1.0 - BER)
#                 tp = 0
#                 fp = 0
#                 fn = 0
#                 tn = 0
#         bintargets=[1,-1]
#         #print features[0][i]
#         #print features[0][i].shape
#         #f=features[0][i,:]
#         #print f.shape
#         #print f.shape
#         #print g_it.target[i]    
#         #third parameter is compulsory just for the first call
#     print "prediction", pred, score
#     #print "intecept",PassiveAggressive.intercept_
#     #raw_input()
#         if abs(score)<1.0 or pred!=g_it.target[i]:
#     
#         ClassWeight=compute_class_weight('auto',np.asarray([1,-1]),bintargets)
#         #print "class weights", {1:ClassWeight[0],-1:ClassWeight[1]}
#         PassiveAggressive.class_weight={1:ClassWeight[0],-1:ClassWeight[1]}
# 
#         PassiveAggressive.partial_fit(ex,np.array([g_it.target[i]]),np.unique(g_it.target))
#         #PassiveAggressive.partial_fit(ex,np.array([g_it.target[i]]),np.unique(g_it.target))
#         W_old=PassiveAggressive.coef_
#         
#         
#         #ESN target---#
#         netTargetSet=[]
#         for key,rowDict in list_for_deep[i].iteritems():
# 
# 
#         target=np.asarray( [np.asarray([W_old[0,key]])]*len(rowDict))
#         
#           
#         netTargetSet.append(target)
# 
# 
#         
# 
#         #------------ESN TargetSetset--------------------#
#         # ESN Training
#         
#         #for ftDataset,ftTargetSet in zip(netDataSet,netTargetSet):
#         #print "Input"
#         #print netDataSet
#         #raw_input("Output")
#         #print netTargetSet
#         #raw_input("Target")
#         model.OnlineTrain(netDataSet,netTargetSet,lr)
#         #raw_input("TR")
#         #calcolo statistiche
# 
# print "BER AVG", sum(BERtotal) / float(len(BERtotal))
# print>>f,"BER AVG "+str(sum(BERtotal) / float(len(BERtotal)))
# f.close()
#==============================================================================
