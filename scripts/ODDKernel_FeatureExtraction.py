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
import numpy as np
import time
from sklearn import cross_validation
from skgraph.feature_extraction.graph import ODDSTVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

if __name__=='__main__':
    if len(sys.argv)<0:
        sys.exit("python ODDKernel_FeatureExtraction.py")
    
    max_radius=4
    la=1.4
    
    #load bursi dataset
    from skgraph import datasets
    dat=datasets.load_graphs_bursi()
    g_it=dat.graphs
    print "Number of examples:",len(dat.graphs)
    y=dat.target

    ODDvec=ODDSTVectorizer.ODDSTVectorizer(r=max_radius,l=la)
    print "Computing features"
    start_time = time.time()
    X=ODDvec.transform(g_it) #instance-featues matrix
    elapsed_time = time.time() - start_time
    print "Took %d s" % (elapsed_time)
    print 'Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1],  X.getnnz()/X.shape[0])
    
    print "Non zero different features %d" % (len(np.unique(X.nonzero()[1])))
    

    #induce a predictive model
    predictor = LinearSVC(n_iter=150,shuffle=True)

    #predictor = SGDClassifier(n_iter=150,shuffle=True)
    print "Training SGD classifier optimizing accuracy"

    scores = cross_validation.cross_val_score(predictor, X, y,cv=10, scoring='accuracy')
    import numpy as np
    print('Accuracy: %.4f +- %.4f' % (np.mean(scores),np.std(scores)))
    print "Training SGD classifier optimizing AUROC"

    scores = cross_validation.cross_val_score(predictor, X, y,cv=10, scoring='roc_auc')
    
    print('AUC ROC: %.4f +- %.4f' % (np.mean(scores),np.std(scores)))
