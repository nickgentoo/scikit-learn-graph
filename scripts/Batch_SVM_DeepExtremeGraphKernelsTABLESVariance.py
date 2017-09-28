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
from sklearn.model_selection import train_test_split
import sys
from skgraph.feature_extraction.graph.ODDSTVectorizer import ODDSTVectorizer
from skgraph.feature_extraction.graph.WLVectorizer import WLVectorizer
import random
from sklearn.linear_model import PassiveAggressiveClassifier as PAC
from skgraph.datasets import load_graph_datasets
import numpy as np
from scipy.sparse import csc_matrix
from sklearn.utils import compute_class_weight
from scipy.sparse import csr_matrix
from skgraph.utils.countminsketch_TABLESrandomprojectionBiasVariance import CountMinSketch
from itertools import izip
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import time
if __name__=='__main__':
    start_time = time.time()

    if len(sys.argv)<1:
        sys.exit("python ODDKernel_example.py dataset r l filename kernel C m rs alpha")
    dataset=sys.argv[1]
    max_radius=int(sys.argv[2])
    la=float(sys.argv[3])
    #hashs=int(sys.argv[3])
    njobs=1
    name=str(sys.argv[4])
    kernel=sys.argv[5]
    C=float(sys.argv[6])
    m=int(sys.argv[7])
    rs=int(sys.argv[8])
    alpha=float(sys.argv[9])

    #lr=float(sys.argv[7])
    #FIXED PARAMETERS
    normalization=False
    #working with Chemical
    g_it=load_graph_datasets.dispatch(dataset)
    epochs=20

    f=open(name,'w')

    from random import shuffle

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
    from sklearn.model_selection import KFold, cross_val_score
    Xind=xrange(features.shape[0])
    random.seed(42)
    k_fold = KFold(n_splits=3, random_state=rs)
    bestaccsval=[0.0]*3
    bestaccstest=[0.0]*3

    fold_index=-1
    for train_val_indices, test_indices in k_fold.split(Xind):
        train_indices, val_indices = train_test_split(train_val_indices, test_size=.20, random_state=0)
        #print train_indices
        #print val_indices

        fold_index+=1
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
        sizes=[5000]*50
        transformer=CountMinSketch(m,features.shape[1],rs)
        #WCMS=np.zeros(shape=(m,1))
        cms_creation=0.0
        clf = LinearSVC(random_state=42,C=C, dual=True,verbose=2)

        #compute the first W
        transformedExamples=[]
        for i in train_indices:
            ex=features[i][0].T
            #print ex

            exCMS=transformer.transform(ex)
            #print "exCMS", exCMS.shape
            transformedExamples.append(np.squeeze(np.asarray(exCMS)))
        #print np.asarray(transformedExamples).shape
        #test=np.array(transformedExamples)
        print "transformedExamples"#, test.shape

        clf.fit(transformedExamples, g_it.target[train_indices])
        print "trained svm"

        #print "W", type(clf.coef_), clf.coef_
        WCMS=clf.coef_

        print "classify Validation set"
        predictionsval = []
        for i in val_indices:
            ex = features[i][0].T

            exCMS = transformer.transform(ex)
            dot = np.dot(WCMS, exCMS)[0, 0]
            pred = np.sign(dot)
            predictionsval.append(pred)
        # print "predictions"
        # todo test on validation set until convergence, than on test set
        accval = accuracy_score(predictionsval, g_it.target[val_indices])
        print "Validation set Accuracy:", accval

        # print "classify Test set"
        predictionstest = []
        for i in test_indices:
            ex = features[i][0].T

            exCMS = transformer.transform(ex)
            dot = np.dot(WCMS, exCMS)[0, 0]
            pred = np.sign(dot)
            predictionstest.append(pred)
        # print "predictions"
        # todo test on validation set until convergence, than on test set
        acctest = accuracy_score(predictionstest, g_it.target[test_indices])
        print "Test set Accuracy:", acctest
        if accval > bestaccsval[fold_index]:
            bestaccsval[fold_index] = accval
            bestaccstest[fold_index] = acctest

        #Now WCMS is fixed, let's learn the representation!


        for e in xrange(epochs):
            print "epoch ", e, "Learning rate C:", C/(e+1), "Learning rate bias:",alpha/(e+1), "Learning rate variance:", np.sqrt(alpha)/(e+1)

            #todo shuffle train indices
            random.seed(e)
            shuffle(train_indices)
            for i in train_indices:
              time1=time.time()

              ex=features[i][0].T

              exCMS=transformer.transform(ex)
              #print "exCMS", type(exCMS), exCMS.shape
              #print exCMS
              target=g_it.target[i]
              #W=csr_matrix(ex)

              #dot=0.0
              module = (np.linalg.norm(np.array(exCMS.T),2)**2)
              #print module
              #module=np.dot(exCMS.T,exCMS)[0,0]
              #print "module", module

              dot=np.dot(WCMS,exCMS)[0,0]
              #print "dot", dot


              tao = min (C, max (0.0,( (1.0 - target*dot )) / module ) );

              if (tao > 0.0):
                    transformer.updatevariance((WCMS.T * target), alpha/(e+1))

                    transformer.updatebias(( WCMS.T*target)  ,alpha/(e+1))

            #re-transform all examples, re-run SVM and classify validation and test
            #TODO
            transformedExamples = []
            for i in train_indices:
                ex = features[i][0].T
                # print ex

                exCMS = transformer.transform(ex)
                transformedExamples.append(np.squeeze(np.asarray(exCMS)))
            # print np.asarray(transformedExamples).shape
            print "transformedTrainingExamples"#, test.shape
            clf = LinearSVC(random_state=0, C=C)

            clf.fit(transformedExamples, g_it.target[train_indices])
            # print "W", type(clf.coef_), clf.coef_
            WCMS = clf.coef_

            #------------

            print "classify Validation set"
            predictionsval=[]
            for i in val_indices:
                ex = features[i][0].T

                exCMS = transformer.transform(ex)
                dot = np.dot(WCMS, exCMS)[0, 0]
                pred=np.sign(dot)
                predictionsval.append(pred)
            #print "predictions"
            #todo test on validation set until convergence, than on test set
            accval= accuracy_score(predictionsval, g_it.target[val_indices])
            print "Validation set Accuracy:", accval

            #print "classify Test set"
            predictionstest=[]
            for i in test_indices:
                ex = features[i][0].T

                exCMS = transformer.transform(ex)
                dot = np.dot(WCMS, exCMS)[0, 0]
                pred=np.sign(dot)
                predictionstest.append(pred)
            #print "predictions"
            #todo test on validation set until convergence, than on test set
            acctest= accuracy_score(predictionstest, g_it.target[test_indices])
            print "Test set Accuracy:", acctest
            if accval > bestaccsval[fold_index]:
                bestaccsval[fold_index]=accval
                bestaccstest[fold_index]=acctest

        transformer.removetmp()

end_time=time.time()
print("Learning phase time %s seconds ---" % (end_time - features_time )) #- cms_creation
print("Total time %s seconds ---" % (end_time - start_time))
print "DEBUG: AVG Accuracy Validation set", str(np.average(bestaccsval)),"std", np.std(bestaccsval)

print "AVG Accuracy Test set", str(np.average(bestaccstest)),"std", np.std(bestaccstest)
#f.write("BER AVG "+ str(np.average(BERtotal))+" std "+str(np.std(BERtotal))+"\n")

f.close()
#transformer.removetmp()

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
#         bintargets.append(target)
#             if pred!=target:
#                 errors+=1
#                 print "Error",errors," on example",i, "pred", score, "target",target
#                 if target==1:
#                     fn+=1
#                 else:
#                     fp+=1
#             
#             else:
#                 if target==1:
#                     tp+=1
#                 else:
#                     tn+=1
#                 #print "Correct prediction example",i, "pred", score, "target",target
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
