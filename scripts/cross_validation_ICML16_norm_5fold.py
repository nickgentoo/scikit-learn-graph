import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '',''))
import numpy as np
#from skgraph import datasets
from sklearn import svm
#from skgraph.ioskgraph import *
from math import sqrt
from copy import copy
import sys
#"sys.path.append('..\\..\\Multiple Kernel Learning\\Framework')"
if len(sys.argv)<4:
    sys.exit("python cross_validation_from_matrix_norm.py inputMatrix.libsvm C outfile")

c=float(sys.argv[2])

##TODO read from libsvm format
from sklearn.datasets import load_svmlight_file
#TODO metodo + veloce per caricar ele amtrici (anche per fare dump)
#from svmlight_loader import load_svmlight_file # riga 22 non serve 
km, target_array = load_svmlight_file(sys.argv[1])
#print type(target_array)
#print target_array
#Controlla se target array ha +1 e -1! se ha 0, sostituisco gli 0 ai -1
if not -1 in target_array:
    print "WARNING: no -1 in target array! Changing 0s to -1s"
    target_array = np.array([-1 if x == 0 else x for x in target_array])
#print km
#tolgo indice
##############kmgood=km[:,1:].todense()
gram=km[:,1:].todense()
kmgood=copy(gram)
#NORMALIZATION
for i in xrange(len(target_array)):
    for j in xrange(0,len(target_array)):
        #print i,j,kmgood[i,j],kmgood[i,i],kmgood[j,j]
	if kmgood[i,i]*kmgood[j,j]==0:
		print "WARNING: avoided divizion by zero"
		gram[i,j]=0
	else:
        	gram[i,j]=kmgood[i,j]/sqrt(kmgood[i,i]*kmgood[j,j])
#-----------------------------------
print "matrix normalization completed"
         
#from sklearn.metrics import make_scorer
# (16) in the paper
def my_custom_loss_func(ground_truth, predictions):
    total_loss=0.0
    for gt,p in zip(ground_truth, predictions):
         #print gt, p
         diff = (1.0 - (gt * p)) / 2.0 
         if diff<0:
             diff=0.0
         if diff > 1.0:
             diff=1.0
         total_loss+=diff
    return total_loss / len(predictions)

from sklearn import cross_validation
import time

start = time.time()
for rs in range(42,43):
#for rs in range(42,53):
    f=open(str(sys.argv[3]+".seed"+str(rs)+".c"+str(c)),'w')

    
    kf = cross_validation.StratifiedKFold(target_array, n_folds=5, shuffle=True,random_state=rs)
    #print kf    
    #remove column zero because
    #first entry of each line is the index
    
    #gram=km[:,1:].todense()
    f.write("Total examples "+str(len(gram))+"\n")
    f.write("|W| \t train_loss \t test_loss\n")
    #print gram
    # normalization
    from math import sqrt
    #for i in range(len(gram)):
    #    for j in range(len(gram)):
    #        gram[i,j]=gram[i,j]/sqrt(gram[i,i]+gram[j,j])
    
    sc=[]
    for train_index, test_index in kf:
        #print("TRAIN:", train_index, "TEST:", test_index)
    
        #generated train and test lists, incuding indices of the examples in training/test
        #for the specific fold. Indices starts from 0 now
        
        clf = svm.SVC(C=c, kernel='precomputed')
        train_gram = [] #[[] for x in xrange(0,len(train))]
        test_gram = []# [[] for x in xrange(0,len(test))]
        #compute training and test sub-matrices
        index=-1    
        for row in gram:
            index+=1
            if index in train_index:
                train_gram.append([gram[index,i] for i in train_index])
            else:
                test_gram.append([gram[index,i] for i in train_index])      
    
        #print gram
        X_train, X_test, y_train, y_test = np.array(train_gram), np.array(test_gram), target_array[train_index], target_array[test_index]
    
        clf.fit(X_train, y_train)
        #print |W|^2= alpha Q alpha, where Q_ij= y_i y_j K(x_i,x_j)
        alpha = clf.dual_coef_ 
        yw=target_array[clf.support_]
        Kw=gram[clf.support_,:][:,clf.support_]
        #print yw.shape, Kw.shape, gram.shape
        yw.shape=(yw.shape[0],1)
        YM=np.ones(yw.shape[0])*yw.T
        Q= np.multiply(np.multiply(YM,Kw),YM.T)
        #print Q.shape
        #print alpha.shape
        #alpha.shape=(alpha.shape[1],1)
        W2=alpha*Q*alpha.T
        print "|W|" , sqrt(W2),
        f.write(str(sqrt(W2))+"\t")

        #loss  = make_scorer(my_custom_loss_func, greater_is_better=False)
    
        from sklearn.metrics import accuracy_score
        #predictions on training set
        y_train_predicted=clf.decision_function(X_train)
        #print type( my_custom_loss_func(y_train, y_train_predicted))
        print " training loss ",(str( my_custom_loss_func(y_train, y_train_predicted))), 
        f.write(str(my_custom_loss_func(y_train, y_train_predicted))+"\t")
        # predict on test examples
        y_test_predicted=clf.decision_function(X_test)
        #print y_test.shape, y_test_predicted.shape
        print " test loss ",(str( my_custom_loss_func(y_test, y_test_predicted)))  
        y_test_predicted_binary=clf.predict(X_test)
        #print y_test
        #print y_test_predicted_binary
        #print "Accuracy: ", accuracy_score(y_test, y_test_predicted_binary)        
        #y_test_sign=map(np.sign, y_test_predicted)
        #print "Accuracy_decision: ", accuracy_score(y_test, y_test_sign)

        sc.append(my_custom_loss_func(y_test, y_test_predicted))
        f.write(str( my_custom_loss_func(y_test, y_test_predicted))+"\n")

    f.close()
end = time.time()
print "Total time:", end-start
scores=np.array(sc)
print "Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() / 2)
    
