__author__ = "Riccardo Tesselli"
__author__ = "Nicolo' Navarin"

import sys
import numpy as np
import time
from skgraph.ioskgraph import load_target
from skgraph import graph
from sklearn import svm
from sklearn import cross_validation
from skgraph.ODDGraphKernel import ODDGraphKernel

if __name__=='__main__':
    if len(sys.argv)<8:
        sys.exit("python ODDKernel_example.py inputDataset inputlabels outfile max_radius lambda_tree hash_size show")
    
    input_data_url=sys.argv[1]
    input_target_file=sys.argv[2]
    out_file=sys.argv[3]
    max_radius=int(sys.argv[4])
    la=float(sys.argv[5])
    hashs=int(sys.argv[6])
    sh=bool(int(sys.argv[7]))
    y=load_target(input_target_file,'file') #labels
    #load data
    from skgraph.graph import graph
    g_it=graph.instance_to_graph(input = input_data_url, input_type = 'file', tool = 'gspan')

    ODDkernel=ODDGraphKernel(r=max_radius,l=la,hashsz=hashs,show=sh)
    print "Computing features"
    start_time = time.time()
    X=ODDkernel.transform(g_it, n_jobs=1) #instance-featues matrix
    elapsed_time = time.time() - start_time
    print "Took %d s" % (elapsed_time)
    print 'Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1],  X.getnnz()/X.shape[0])
    
    print "Non zero different features %d" % (len(np.unique(X.nonzero()[1])))
    
    print "Computing Gram matrix"
    start_time = time.time()
    gram=ODDkernel.computeGram(X)
    elapsed_time = time.time() - start_time
    print "Took %d s" % (elapsed_time)
    
    print "Writing on disk Gram matrix"
    start_time = time.time()
    np.savetxt(out_file, gram,fmt='%1.6f')
    elapsed_time = time.time() - start_time
    print "Took %d s" % (elapsed_time)
    
    #Learner
    clf = svm.SVC(C=10,kernel='precomputed')
    clf.fit(gram, y)
    scores = cross_validation.cross_val_score(clf, gram, y, cv=10)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    """
    #induce a predictive model
    from sklearn.linear_model import SGDClassifier
    predictor = SGDClassifier()
    
    from sklearn import cross_validation
    scores = cross_validation.cross_val_score(predictor, X, y,cv=10, scoring='accuracy')
    
    import numpy as np
    print('Accuracy: %.4f +- %.4f' % (np.mean(scores),np.std(scores)))
    
    scores = cross_validation.cross_val_score(predictor, X, y,cv=10, scoring='roc_auc')
    
    print('AUC ROC: %.4f +- %.4f' % (np.mean(scores),np.std(scores)))
    """