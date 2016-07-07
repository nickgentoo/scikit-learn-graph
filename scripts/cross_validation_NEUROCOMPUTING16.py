import sys, os
from math import sqrt
from copy import copy

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '',''))
import numpy as np
#from skgraph import datasets
from sklearn import svm
#from skgraph.ioskgraph import *
from math import sqrt, ceil
import sys
#"sys.path.append('..\\..\\Multiple Kernel Learning\\Framework')"
if len(sys.argv)<4:
    sys.exit("python cross_validation_from_matrix_norm.py inputMatrix.libsvm C outfile MCit")

c=float(sys.argv[2])
MC=int(sys.argv[4])

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
#NORMALIZATION
#for i in xrange(len(target_array)):
#    for j in xrange(0,len(target_array)):
#        #print i,j,kmgood[i,j],kmgood[i,i],kmgood[j,j]
#	if kmgood[i,i]*kmgood[j,j]==0:
#		print "WARNING: avoided divizion by zero"
#		gram[i,j]=0
#	else:
#        	gram[i,j]=kmgood[i,j]/sqrt(kmgood[i,i]*kmgood[j,j])
#-----------------------------------
#print gram
         
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

X=range(len(gram))
from sklearn import cross_validation
#for rs in range(42,43):
#for rs in range(42,53):
f=open(str(sys.argv[3]+".c"+str(c)),'w')

#NEW CODE FOR SUBSAMPLING
#REPEAT N TIMES
#gram=km[:,1:].todense()
f.write("Total examples "+str(len(gram))+"\n")
f.write("# \t Stability_MCsample\n")
from sklearn.cross_validation import train_test_split
Complexity=0.0
Wtot=0.0
for MCit in xrange(MC):
    #print gram.shape
    print("number of examples "+str(ceil(sqrt(gram.shape[0]))))
    radn=int(ceil(sqrt(gram.shape[0])))
    #print radn
    y_train=[]
    #continue only if both class labels are in the set
    rand=MCit
    while (len(np.unique(y_train))<2):
        train_index, test_index, y_train, y_test = train_test_split(X, target_array, train_size=(radn-1),test_size=1, random_state=rand)
        rand=MCit+MC
    #print train_index, test_index, y_train.shape, y_test.shape 
    #At this point, X_train, X_test are the list of indices to consider in training/test

    sc=[]
    #print("TRAIN:", train_index, "TEST:", test_index)

    #generated train and test lists, incuding indices of the examples in training/test
    #for the specific fold. Indices starts from 0 now
    total_index=copy(train_index)
    total_index.extend(test_index)
    #print total_index
    #print y_train.tolist(),y_test.tolist()
    temp=copy(y_train.tolist())
    temp.extend(y_test.tolist())
    y_total=np.array(temp)
    #y_total.extend(y_test)
    clf = svm.SVC(C=c, kernel='precomputed',max_iter=10000000)
    clf1 = svm.SVC(C=c, kernel='precomputed',max_iter=10000000)
    
    train_gram = [] #[[] for x in xrange(0,len(train))]
    test_gram = []# [[] for x in xrange(0,len(test))]
    total_gram = []
    #compute training and test sub-matrices
    index=-1    
    for row in gram:
        index+=1
        if index in train_index:
            train_gram.append([gram[index,i] for i in train_index])
        elif index in test_index:
            test_gram.append([gram[index,i] for i in train_index]) 
        if index in total_index:    
            total_gram.append([gram[index,i] for i in total_index])   
        #if not in training nor test, just discard the row   

    #print gram
    #X_train, X_test, y_train, y_test = np.array(train_gram), np.array(test_gram),      target_array[train_index], target_array[test_index]
    X_train, X_test = np.array(train_gram), np.array(test_gram)
    X_total=np.array(total_gram) 
    print("Fitting first SVM(training only)")
    clf.fit(X_train, y_train)
    print("Fitting second SVM(training+test)")  
    clf1.fit(X_total,y_total)
    print("Training done")  
    #commented code to compute |W|
    #print |W|^2= alpha Q alpha, where Q_ij= y_i y_j K(x_i,x_j)
    alpha = clf1.dual_coef_ 
    yw=target_array[clf1.support_]
    Kw=gram[clf1.support_,:][:,clf1.support_]
    #print yw.shape, Kw.shape, gram.shape
    yw.shape=(yw.shape[0],1)
    YM=np.ones(yw.shape[0])*yw.T
    Q= np.multiply(np.multiply(YM,Kw),YM.T)
    #print Q.shape
    #print alpha.shape
    #alpha.shape=(alpha.shape[1],1)
    W2=alpha*Q*alpha.T
    print "|W|" , sqrt(W2),
    #f.write("|W| "+str(sqrt(W2))+"\n")
    Wtot+=float(W2)
    #-------------------------

    #loss  = make_scorer(my_custom_loss_func, greater_is_better=False)

    #from sklearn.metrics import accuracy_score
    #predictions on training set
    y_test_predicted=clf.decision_function(X_test)
    #print type( my_custom_loss_func(y_train, y_train_predicted))
    # predict on test examples
    loss_training=my_custom_loss_func(y_test, y_test_predicted)
    y_test_predicted_total=clf1.predict(X_total)
    loss_total=my_custom_loss_func(y_test, y_test_predicted_total)
    Complexity+=abs(loss_training-loss_total)
    print "Complexity", abs(loss_training-loss_total)        
    f.write(str(MCit)+" "+str(abs(loss_training-loss_total))+"\n")
    
    #print y_test.shape, y_test_predicted.shape
    #print y_test
    #print y_test_predicted_binary
    #print "Accuracy: ", accuracy_score(y_test, y_test_predicted_binary)        
    #y_test_sign=map(np.sign, y_test_predicted)
    #print "Accuracy_decision: ", accuracy_score(y_test, y_test_sign)

Complexity/=MC 
Wtot/=MC
f.write(str(abs(loss_training-loss_total))+"\n")
f.write("Stability with "+str(MC)+" MonteCarlo samples: "+str(Complexity)+" Wmax "+str(Wtot)+"\n")
print "Stability with", MC, "MonteCarlo samples:", Complexity,"Wmax", str(Wtot)
f.close()

