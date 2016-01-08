"""
@author: Michele Donini
@email: mdonini@math.unipd.it

EasyMKL: a scalable multiple kernel learning algorithm
by Fabio Aiolli and Michele Donini

Paper @ http://www.math.unipd.it/~mdonini/publications.html
"""

from cvxopt import spmatrix, sparse, matrix, solvers, mul
import numpy as np
import time


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

#        start = time.clock()

        k = spmatrix(0.0,(list_K[0].size[0],list_K[0].size[1]))
        if weights == None:
            for ker in list_K:
                k += ker
        else:
            for w, ker in zip(weights,list_K):
                k += w * ker            

#        end = time.clock()
#        print "[sum_kernel] Elapsed time:", (end-start)

        return k
    
    def traceN(self, k):
#        start = time.clock()
        tn = sum([k[i,i] for i in range(k.size[0])]) / k.size[0]
#        end = time.clock()
#        print "[tranceN] Elapsed time:", (end-start)
        return tn 
    
    def train(self, list_Ktr, labels):
        ''' 
            list_Ktr : list of kernels of the training examples
            labels : array of the labels of the training examples
        '''
        self.list_Ktr = list_Ktr  
        for k in self.list_Ktr:
            self.traces.append(self.traceN(k))
        if self.tracenorm:
            self.list_Ktr = [k / self.traceN(k) for k in list_Ktr]

        set_labels = set(labels)
        if len(set_labels) != 2:
            print 'The different labels are not 2'
            return None
        elif (-1 in set_labels and 1 in set_labels):
            self.labels = labels
        else:
            poslab = np.max(set_labels)
            self.labels = np.array([1 if i==poslab else -1 for i in labels])
        
#        start = time.clock()
        # Sum of the kernels
        ker_matrix = spmatrix(self.sum_kernels(self.list_Ktr))

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

#        end = time.clock()
#        print "[train_solve_1] Elapsed time:", (end-start)
        
#        start = time.clock()
        # Bias for classification:
        bias = 0.5 * self.gamma.T * ker_matrix * YY * self.gamma
        self.bias = bias
#        end = time.clock()
#        print "[train_bias] Elapsed time:", (end-start)

#        start = time.clock()
        # Weights evaluation:
        yg =  mul(self.gamma.T,self.labels.T)
        self.weights = []
        for kermat in self.list_Ktr:
            b = yg*kermat*yg.T
            self.weights.append(b[0])
#        end = time.clock()
#        print "[train_weights] Elapsed time:", (end-start)
            
#        start = time.clock()
        norm2 = sum([w for w in self.weights])
        self.weights = [w / norm2 for w in self.weights]
#        end = time.clock()
#        print "[train_norm] Elapsed time:", (end-start)

#        start = time.clock()
        if self.tracenorm: 
            for idx,val in enumerate(self.traces):
                self.weights[idx] = self.weights[idx] / val        
#        end = time.clock()
#        print "[train_tracenorm] Elapsed time:", (end-start)
        
#        start = time.clock()
        if True:
            ker_matrix = spmatrix(self.sum_kernels(list_Ktr, self.weights))
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
#        end = time.clock()
#        print "[train_solve_2] Elapsed time:", (end-start)
        
        
        return self
    
    def rank(self,list_Ktest):
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
        ker_matrix = spmatrix(self.sum_kernels(list_Ktest, self.weights))
        z = ker_matrix*YY*self.gamma
        return z
