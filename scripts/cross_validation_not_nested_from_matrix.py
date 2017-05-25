import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '',''))
import numpy as np
from skgraph import datasets
from sklearn import svm
from skgraph.datasets import ioskgraph
import sys
#"sys.path.append('..\\..\\Multiple Kernel Learning\\Framework')"
if len(sys.argv)<4:
    sys.exit("python cross_validation_from_matrix.py inputMatrix.libsvm C outfile")

c=float(sys.argv[2])

##TODO read from libsvm format
from sklearn.datasets import load_svmlight_file
km, target_array = load_svmlight_file(sys.argv[1])


from sklearn import cross_validation
for rs in range(42,43):
    f=open(str(sys.argv[3]+".seed"+str(rs)+".c"+str(c)),'w')

    
    kf = cross_validation.StratifiedKFold(target_array, n_folds=10, shuffle=True,random_state=rs)
    #print kf    
    #remove column zero because
    #first entry of each line is the index
    
    gram=km[:,1:].todense()
    f.write("Total examples "+str(len(gram))+"\n")
    f.write("seed\t CV_test_acc\t std\n")

    #print gram
    # normalization
    from math import sqrt
    #for i in range(len(gram)):
    #    for j in range(len(gram)):
    #        gram[i,j]=gram[i,j]/sqrt(gram[i,i]+gram[j,j])
    clf = svm.SVC(C=c, kernel='precomputed')
    scores= cross_validation.cross_val_score(clf, gram, target_array, cv=kf)
        #print "inner scores", inner_scores
    print "CV Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())
    f.write(str(rs)+" "+str(scores.mean())+"\t"+str(scores.std()))

