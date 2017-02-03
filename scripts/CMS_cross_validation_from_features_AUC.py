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
from skgraph.utils.countminsketch import CountMinSketch

#"sys.path.append('..\\..\\Multiple Kernel Learning\\Framework')"
if len(sys.argv)<4:
    sys.exit("python cross_validation_from_matrix_norm.py features.libsvm C outfile m d [class_weight:auto]")

c=float(sys.argv[2])
m=int(sys.argv[4])
d=int(sys.argv[5])
auto_weight=False
if len(sys.argv)==7:
    if sys.argv[6]=="auto":
     auto_weight=True
##TODO read from libsvm format
from sklearn.datasets import load_svmlight_file
features, target_array = load_svmlight_file(sys.argv[1])
#print km
print "original shape", features.shape
print "features loaded, hashing..."
featuresCMS=[]
for i in xrange(features.shape[0]):
          exCMS=CountMinSketch(m,d)

          ex=features[i][0]
          #W=csr_matrix(ex)

          rows,cols = ex.nonzero()
          #dot=0.0
          module=0.0
          for row,col in zip(rows,cols):
              #((row,col), ex[row,col])
              value=ex[row,col]
              #print col, ex[row,col]
              #dot+=WCMS[col]*ex[row,col]
              exCMS.add(col,value)
              #print dot
              #TODO aggiungere bias
          featuresCMS.append(exCMS.asarray())
print "hashing done"
features=np.matrix(featuresCMS)
print features.shape
print features[i].shape
#from sklearn import cross_validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

for rs in range(42,53):
    f=open(str(sys.argv[3]+".seed"+str(rs)+".c"+str(c)),'w')

    
    kf = StratifiedKFold(n_splits=10, shuffle=True,random_state=rs)
    #print kf    
    #remove column zero because
    #first entry of each line is the index
    
    #gram=km[:,1:].todense()
    f.write("Total examples "+str(features.shape[0])+"\n")
    f.write("CV\t test_AUROC\n")

    
    sc=[]
    for train_index, test_index in kf.split(features,target_array):
        #print("TRAIN:", train_index, "TEST:", test_index)
    
        #generated train and test lists, incuding indices of the examples in training/test
        #for the specific fold. Indices starts from 0 now
        if auto_weight==False:
            clf = svm.LinearSVC(C=c,dual=True) #, class_weight='auto'
        else:
            print "Class weights automatically assigned from training data"
            clf = svm.LinearSVC(C=c,dual=True, class_weight='balanced')

            
        #clf = svm.SVC(C=c,probability=True, class_weight='auto',kernel='linear') #,probability=True,
        #clf = linear_model.LogisticRegression(C=c, dual=True, class_weight='auto')#, solver='liblinear'
        #generate train features and test features

        X_train, X_test, y_train, y_test = features[train_index], features[test_index], target_array[train_index], target_array[test_index]
        #COMPUTE INNERKFOLD
        kf = StratifiedKFold(n_splits=10, shuffle=True,random_state=rs)
        inner_scores= cross_val_score(
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
    
