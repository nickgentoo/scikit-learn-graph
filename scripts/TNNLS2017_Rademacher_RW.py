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
if len(sys.argv)<2:
    sys.exit("python cross_validation_from_matrix_norm.py inputMatrix.libsvm")


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

#emulate rademacher computation
eigenvalues=np.linalg.eigvalsh(gram) #for symmetric matrices, already ordered

r = np.linspace(0, 1, 30);
LR = []
for j in r:
    tmp=[e for e in eigenvalues if i >= j]
    LR.append(sum(tmp))
dio = np.trapz(LR,r)
retto = max(LR) * max(r)
radem = dio / retto
print "Rademacher", radem
end = time.time()
print "Rademacher time:", end-start


start = time.time()
for c in np.logspace(-11, 5, num=17, endpoint=True, base=10.0):

    clf = svm.SVC(C=c, kernel='precomputed')
    X_train, y_train = gram, target_array
    clf.fit(X_train, y_train)
end = time.time()
print "Single training total time (19 C values):", end-start
#scores=np.array(sc)
#print "Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() / 2)
    
