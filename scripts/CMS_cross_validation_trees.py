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
from skgraph.datasets.load_tree_datasets import dispatch
import networkx
import matplotlib.pyplot as plt
import multiprocessing
from skgraph.feature_extraction.graph.ODDSTVectorizer import ODDSTVectorizer
from skgraph.feature_extraction.graph.NSPDK.NSPDKVectorizer import NSPDKVectorizer
from skgraph.feature_extraction.graph.WLVectorizer import WLVectorizer
#"sys.path.append('..\\..\\Multiple Kernel Learning\\Framework')"
if len(sys.argv)<4:
    sys.exit("python cross_validation_from_matrix_norm.py dataset kernel C outfile m d [class_weight:auto]")
kernel=sys.argv[2]
c=float(sys.argv[3])
m=int(sys.argv[5])
d=int(sys.argv[6])
auto_weight=False
if len(sys.argv)==7:
    if sys.argv[6]=="auto":
     auto_weight=True
##TODO read from libsvm format

max_radius=3
normalization=False

def generateCMS(featuresCMS,ex,i):
    exCMS = CountMinSketch(m, d)
    # W=csr_matrix(ex)

    rows, cols = ex.nonzero()
    # dot=0.0
    for row, col in zip(rows, cols):
        # ((row,col), ex[row,col])
        value = ex[row, col]
        # print col, ex[row,col]
        # dot+=WCMS[col]*ex[row,col]
        exCMS.add(col, value)
        # print dot
        # TODO aggiungere bias
    featuresCMS[i]=exCMS.asarray()

g_it=dispatch(sys.argv[1])
# graph=g_it.graphs[0]
# node_positions = networkx.spring_layout(graph)
# networkx.draw_networkx_nodes(graph, node_positions, with_labels = True)
# networkx.draw_networkx_edges(graph, node_positions, style='dashed')
# plt.show()
if kernel == "WL":
    print "Lambda ignored"
    print "Using WL fast subtree kernel"
    Vectorizer = WLVectorizer(r=max_radius, normalization=normalization)
elif kernel == "ODDST":
    print "Using ST kernel"
    Vectorizer = ODDSTVectorizer(r=max_radius, l=la, normalization=normalization)
elif kernel == "NSPDK":
    print "Using NSPDK kernel, lambda parameter interpreted as d"
    Vectorizer = NSPDKVectorizer(r=max_radius, d=int(la), normalization=normalization)
else:
    print "Unrecognized kernel"

features = Vectorizer.transform(g_it.graphs)
target_array=np.array(g_it.target)
#features, target_array =
#print km
print "original shape", features.shape
print "features loaded, hashing..."
featuresCMS=[0]*features.shape[0]
for i in xrange(features.shape[0]):
          generateCMS(featuresCMS,features[i][0],i)
          #pool = multiprocessing.Pool(processes=4)
          #pool.map(generateCMS, features[i][0],i)
          # pool.close()
          # pool.join()

          # exCMS=CountMinSketch(m,d)
          #
          # ex=features[i][0]
          # #W=csr_matrix(ex)
          #
          # rows,cols = ex.nonzero()
          # #dot=0.0
          # for row,col in zip(rows,cols):
          #     #((row,col), ex[row,col])
          #     value=ex[row,col]
          #     #print col, ex[row,col]
          #     #dot+=WCMS[col]*ex[row,col]
          #     exCMS.add(col,value)
          #     #print dot
          #     #TODO aggiungere bias
          # featuresCMS.append(exCMS.asarray())
print "hashing done"
features=np.matrix(featuresCMS)
print features.shape
print features[i].shape
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
        if auto_weight==False:
            clf = svm.LinearSVC(C=c,dual=True) #, class_weight='auto'
        else:
            print "Class weights automatically assigned from training data"
            clf = svm.LinearSVC(C=c,dual=True, class_weight='auto')

            
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
    
