import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '',''))
from pythontk import tree
from pythontk import tree_kernels
import numpy as np
from skgraph import datasets
from sklearn import svm
from ioskgraph import *
import sys
#"sys.path.append('..\\..\\Multiple Kernel Learning\\Framework')"
if len(sys.argv)<4:
    sys.exit("python prova2.py inputMatrix  CVIndices C")

dat = tree_kernels.KernelMatrixLibsvm.loadFromLIBSVMFile(sys.argv[1])
print dat
c=float(sys.argv[3])
#indices=load_target(sys.argv[3], input_type = 'file')
#print indices

#generate indices from the shitty format. Inidces starts from 1 (double shitty format)
ind=open(sys.argv[2],'r')
indices={}
fold=0
for line in ind :
    fold=fold+1
    for word in line.split():
        if (word.find('#')==-1):
            indices[int(word)]=fold
#indices[example] tells on which fold the example is
sc=[]
#for each fold
for fold in range(1,11): # last index excluded
    print 'processing fold '+str(fold)
    train=[]
    test=[]
    for i in indices:
        if (indices[i]==fold):
            test.append(i-1)
        else:
            train.append(i-1)
    #generated train and test lists, incuding indices of the examples in training/test
    #for the specific fold. Indices starts from 0 now
    
    clf = svm.SVC(C=c, kernel='precomputed')
    #print dat.km[0]
    gram = [[] for x in xrange(0,len(dat.km))]
    train_gram = [] #[[] for x in xrange(0,len(train))]
    test_gram = []# [[] for x in xrange(0,len(test))]
    r=-1
    for row in dat.km:
        r+=1
        gram[r]=[0 for i in range(0,len(dat.km))]           
        for key,element in row.iteritems():
            if key>0: # in libsvm format, the 0 index is the element id
                gram[r][key-1]=element
    #at this point, gram is a decent gram matrix
                
    #generate train matrix and test matrix
    index=-1    
    for row in gram:
        index+=1
        if index in train:
            train_gram.append([gram[index][i] for i in train])
        else:
            test_gram.append([gram[index][i] for i in train])



    #print gram
    target_array=np.array(dat.target)
    X_train, X_test, y_train, y_test = np.array(train_gram), np.array(test_gram), target_array[train], target_array[test]

    print clf.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    # predict on test examples
    y_test_predicted=clf.predict(X_test)
    sc.append(accuracy_score(y_test, y_test_predicted))
scores=np.array(sc)
print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

