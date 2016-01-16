"""
@author: Michele Donini
@email: mdonini@math.unipd.it

EasyMKL: a scalable multiple kernel learning algorithm
by Fabio Aiolli and Michele Donini

Paper @ http://www.math.unipd.it/~mdonini/publications.html
"""

from cvxopt import matrix, solvers, mul
import numpy as np


class EasyMKL():
    ''' EasyMKL is a Multiple Kernel Learning algorithm.
        The parameter lam (lambda) has to be validated from 0 to 1.

        For more information:
        EasyMKL: a scalable multiple kernel learning algorithm
            by Fabio Aiolli and Michele Donini

        Paper @ http://www.math.unipd.it/~mdonini/publications.html
    '''
    def __init__(self, lam = 0.1, tracenorm = True):
        self.lam = lam
        self.tracenorm = tracenorm
        
        self.list_Ktr = None
        self.labels = None
        self.gamma = None
        self.weights = None
        self.traces = []

    def sum_kernels(self, list_K, weights = None):
        ''' Returns the kernel created by averaging of all the kernels '''
        k = matrix(0.0,(list_K[0].size[0],list_K[0].size[1]))
        if weights == None:
            for ker in list_K:
                k += ker
        else:
            for w,ker in zip(weights,list_K):
                k += w * ker            
        return k
    
    def traceN(self, k):
        return sum([k[i,i] for i in range(k.size[0])]) / k.size[0]
    
    def train(self, sum_Ktr, labels):
        ''' 
            sum_Ktr :  a single kernel of the training examples (the sum of all the kernels)
            labels : array of the labels of the training examples
        '''
        self.sum_Ktr = sum_Ktr

        set_labels = set(labels)
        if len(set_labels) != 2:
            print 'The different labels are not 2'
            return None
        elif (-1 in set_labels and 1 in set_labels):
            self.labels = labels
        else:
            poslab = np.max(set_labels)
            self.labels = np.array([1 if i==poslab else -1 for i in labels])
        
        # Sum of the kernels
        ker_matrix = matrix(self.sum_Ktr)

        YY = matrix(np.diag(list(matrix(labels))))
        KLL = (1.0-self.lam)*YY*ker_matrix*YY
        LID = matrix(np.diag([self.lam]*len(labels)))
        Q = 2*(KLL+LID)
        p = matrix([0.0]*len(labels))
        G = -matrix(np.diag([1.0]*len(labels)))
        h = matrix([0.0]*len(labels),(len(labels),1))
        A = matrix([[1.0 if lab==+1 else 0 for lab in labels],[1.0 if lab2==-1 else 0 for lab2 in labels]]).T
        b = matrix([[1.0],[1.0]],(2,1))
        
        solvers.options['show_progress']=False#True
        sol = solvers.qp(Q,p,G,h,A,b)
        # Gamma:
        self.gamma = sol['x']     
        
        # Bias for classification:
        bias = 0.5 * self.gamma.T * ker_matrix * YY * self.gamma
        self.bias = bias

        return self

    def train2(self, sum_Ktr, labels):
        ker_matrix = matrix(sum_Ktr)
        YY = matrix(np.diag(list(matrix(labels))))
        
        KLL = (1.0-self.lam)*YY*ker_matrix*YY
        LID = matrix(np.diag([self.lam]*len(labels)))
        Q = 2*(KLL+LID)
        p = matrix([0.0]*len(labels))
        G = -matrix(np.diag([1.0]*len(labels)))
        h = matrix([0.0]*len(labels),(len(labels),1))
        A = matrix([[1.0 if lab==+1 else 0 for lab in labels],[1.0 if lab2==-1 else 0 for lab2 in labels]]).T
        b = matrix([[1.0],[1.0]],(2,1))
        
        solvers.options['show_progress']=False#True
        sol = solvers.qp(Q,p,G,h,A,b)
        # Gamma:
        self.gamma = sol['x']
    
    
        return self
    
    def rank(self, sum_Kte):
        '''
            list_Ktr : list of kernels of the training examples
            labels : array of the labels of the training examples
            Returns the list of the examples in test set of the kernel K ranked
        '''
        if self.weights == None:
            print 'EasyMKL has to be trained first!'
            return
         
        #YY = matrix(np.diag(self.labels).copy())
        YY = matrix(np.diag(list(matrix(self.labels))))
        ker_matrix = matrix(sum_Kte)
        z = ker_matrix*YY*self.gamma
        return z
