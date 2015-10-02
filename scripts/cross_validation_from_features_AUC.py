import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '',''))
import numpy as np
#from skgraph import datasets
from sklearn import svm
#from skgraph.ioskgraph import *
from math import sqrt
import sys
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
from sklearn import linear_model 

#"sys.path.append('..\\..\\Multiple Kernel Learning\\Framework')"
if len(sys.argv)<4:
    sys.exit("python cross_validation_from_matrix_norm.py features.libsvm C outfile")

c=float(sys.argv[2])

##TODO read from libsvm format
from sklearn.datasets import load_svmlight_file
features, target_array = load_svmlight_file(sys.argv[1])
#print km

from sklearn import cross_validation
for rs in range(42,53):
    f=open(str(sys.argv[3]+".seed"+str(rs)+".c"+str(c)),'w')

    
    kf = cross_validation.StratifiedKFold(target_array, n_folds=10, shuffle=True,random_state=rs)
    #print kf    
    #remove column zero because
    #first entry of each line is the index
    
    #gram=km[:,1:].todense()
    f.write("Total examples "+str(features.shape[0])+"\n")
    f.write("CV\t test_AUROC\n")

    
    sc=[]
    for train_index, test_index in kf:
        #print("TRAIN:", train_index, "TEST:", test_index)
    
        #generated train and test lists, incuding indices of the examples in training/test
        #for the specific fold. Indices starts from 0 now
        clf = svm.LinearSVC(C=c,dual=True) #, class_weight='auto'
        #clf = svm.SVC(C=c,probability=True, class_weight='auto',kernel='linear') #,probability=True,
        #clf = linear_model.LogisticRegression(C=c, dual=True, class_weight='auto')#, solver='liblinear'
        #generate train features and test features

        X_train, X_test, y_train, y_test = features[train_index], features[test_index], target_array[train_index], target_array[test_index]
        #COMPUTE INNERKFOLD
        kf = cross_validation.StratifiedKFold(y_train, n_folds=10, shuffle=True,random_state=rs)
        inner_scores= cross_validation.cross_val_score(
        clf, X_train, y_train, cv=kf, scoring='roc_auc')
        #print "inner scores", inner_scores
        print "Inner AUROC: %0.4f (+/- %0.4f)" % (inner_scores.mean(), inner_scores.std() / 2)

        f.write(str(inner_scores.mean())+"\t")

    
        clf.fit(X_train, y_train)
    
        # predict on test examples
        #LibLinear does not support multiclass
        y_test_predicted=clf.decision_function(X_test)
        #y_test_predicted=clf.predict_proba(X_test)
#        #print y_test_predicted
#        sc.append(roc_auc_score(y_test, y_test_predicted[:,1]))
#        f.write(str(roc_auc_score(y_test, y_test_predicted[:,1]))+"\n")
        #LibLinear does not support multiclass
        #print y_test_predicted
        sc.append(roc_auc_score(y_test, y_test_predicted))
        f.write(str(roc_auc_score(y_test, y_test_predicted))+"\n")


    f.close()
scores=np.array(sc)
print "AUROC: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() / 2)
    
