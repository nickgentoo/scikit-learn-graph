# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:23:52 2016

Copyright 2015 Nicolo' Navarin

This file is part of count-mean-sketch based on https://github.com/rafacarrascosa/countminsketch.

count-mean-sketch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

count-mean-sketch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with count-mean-sketch.  If not, see <http://www.gnu.org/licenses/>.
"""
import hashlib
import os
#import array
import itertools
import numpy as np
import string
from numpy import median
import numpy.matlib
import copy
from itertools import izip
from numpy import random, sqrt, log, sin, cos, pi
from scipy.sparse import csr_matrix, linalg
#from joblib import Parallel, delayed
#import multiprocessing
import scipy
import tables as tb
def processInput(i, m, rs):
    numpy.random.seed(i + (rs * 10000))

    v = numpy.random.normal(0, 1, m)
    v = numpy.multiply(sqrt(m), v)
    row = [idx for idx in xrange(m)]
    col = [i for idx in xrange(m)]
    data = v
    return (row, col, data)

class CountMinSketch(object):
    """
    A class for counting hashable items using the Count-min Sketch strategy.
    It fulfills a similar purpose than `itertools.Counter`.

    The Count-min Sketch is a randomized data structure that uses a constant
    amount of memory and has constant insertion and lookup times at the cost
    of an arbitrarily small overestimation of the counts.

    It has two parameters:
     - `m` the size of the hash tables, larger implies smaller overestimation
     - `d` the amount of hash tables, larger implies lower probability of
           overestimation.

    An example usage:

        from countminsketch import CountMinSketch
        sketch = CountMinSketch(1000, 10)  # m=1000, d=10
        sketch.add("oh yeah")
        sketch.add(tuple())
        sketch.add(1, value=123)
        print sketch["oh yeah"]       # prints 1
        print sketch[tuple()]         # prints 1
        print sketch[1]               # prints 123
        print sketch["non-existent"]  # prints 0

    Note that this class can be used to count *any* hashable type, so it's
    possible to "count apples" and then "ask for oranges". Validation is up to
    the user.
    """
    def __init__(self, m, samplesize,rs):
        """ sizes is an array of hash dimensions.
        """
        if not m:
            raise ValueError("Table size (m) and amount of hash functions (d)"
                             " must be non-zero")

        self.n = 0
        self.m=m
        self.samplesize=samplesize
        self.rs=rs
        self.mus=numpy.asarray([0.0] *m).reshape(self.m,1)
        print "mus", self.mus.shape
        #self.tables = numpy.matlib.zeros(shape=(m,samplesize))
        #self.tables=numpy.random.normal(size=(m,samplesize))
#        for _ in xrange(d):
#            table = array.array("d", (0.0 for _ in xrange(m)))
#            self.tables.append(table)
        #inizialize projection matrix
        import random as rnd

        #numpy.random.seed(self.rs * 10000)
        filename=''.join(rnd.choice(string.ascii_uppercase + string.digits) for _ in range(16))
        #filename= "test"
        self.filename=filename+'.h5'

        h5file = tb.open_file(self.filename, mode='w', title="Random Projection Matrix")
        root = h5file.root
        self.x = h5file.create_carray(root, 'x', tb.Float64Atom(), shape=(self.samplesize, self.m))
        print "generating matrix of shape", self.samplesize, self.m
        for i in range(self.samplesize):
            numpy.random.seed(i + (self.rs * 10000))
            #v = numpy.random.normal(0, 1, self.m)
            self.x[i, :self.m] = numpy.random.normal(0, 1, self.m)  # Now put in some data
        print "Random projection matrix saved on file", filename+'.h5'

    def transform(self, vector):
        #mus is a vector of the means
        #print "example size", vector.shape
        #print "transformation size", self.tables.shape
        #tables=csr_matrix ((self.m,self.samplesize))

        #num_cores = multiprocessing.cpu_count()
        indices=vector.nonzero()[0]
        #print vector.shape
        norm=scipy.sparse.linalg.norm(vector,1)
        #print norm

        # results = Parallel(n_jobs=num_cores)(delayed(processInput)(i,self.m,self.rs) for i in indices)
        # parrow = []
        # parcol = []
        # pardata = []
        # for (row,col,v) in results:
        #     parrow.extend(row)
        #     parcol.extend(col)
        #     pardata.extend(v)


        row=[]
        col=[]
        data=[]
        data_nobias=[]
        vbias=[]

        #print indices
        #print indices
        #RPM=self.x[indices,:self.m]
        #print RPM
        data_nobias=self.x[indices,:self.m].ravel()

        #data_nobias=list(itertools.chain.from_iterable([self.x[i,:self.m] for i in indices]))
        #print data_nobias
        data=np.tile(numpy.multiply(norm, self.mus).ravel(),len(indices))
        #data=list(itertools.chain.from_iterable([numpy.multiply(norm, self.mus).ravel()]*len(indices)))
        #print data
        row=np.tile(range(self.m),len(indices))
        #row=range(self.m)*len(indices)
        #print row
        col=np.repeat(indices, self.m)
        #col=np.tile([i]* self.m,len(indices))
        #col=list(itertools.chain.from_iterable([[i]* self.m for i in indices]))
        #print col

        # print data_nobias
        # for i in indices:
        #     #numpy.random.seed(i+(self.rs*10000))
        #     v=self.x[i,:self.m].reshape(self.m,1)
        #     #v=numpy.multiply(sqrt(self.m),v).reshape(self.m,1)
        #     #print "v", v.shape
        #     #print "munorm", (self.mus*norm).shape
        #     #vbias.extend(numpy.multiply(norm, self.mu))
        #     #print "vbias", vbias.shape
        #     row.extend(range(self.m))
        #     col.extend([i]* self.m)
        #     data.extend(numpy.multiply(norm, self.mus).ravel()) #considero il bias
        #     data_nobias.extend(v.ravel())
            #print data
        tables_nobias=csr_matrix ((data_nobias,(row,col)), shape=(self.m,self.samplesize))
        tables_nobias=scipy.sparse.csr_matrix.multiply(tables_nobias,sqrt(self.m))

        #vbias.extend(numpy.multiply(norm,self.mu))
        toadd=csr_matrix ((data,(row,col)), shape=(self.m,self.samplesize))
        tables=tables_nobias+ toadd #csr_matrix ((data,(row,col)), shape=(self.m,self.samplesize))
        transformation= numpy.tanh(np.multiply(tables,vector)).todense()
        #print transformation.shape
        #assert(parrow==row)
        #assert(parcol==col)
        #assert(pardata==data)
        #TODO return vector in which i-th (1-tanh(R_i\phi(g) +norm*\mu_i)^2 * norm)
        #then just multiply each entry by y w_i to get the gradient
        self.norm=norm
        val2= self.norm*self.mus
        #print "val2", val2.shape
        #print "tablesnobias", tables_nobias.shape
        #print "vector", vector.shape
        self.Rphix= (np.multiply(tables_nobias,vector)).todense()
        val3=self.Rphix+val2
        #print "val3",val3.shape
        ones = np.ones(self.m).reshape(self.m,1)
        #print "ones", ones.shape
        derivative= np.multiply((ones-numpy.square(val3)),norm)
        #print derivative

        return transformation # Probably I'll need to return v (to compute the bs)


    def removetmp(self):
        os.remove(self.filename)
        print "removed temporary file"

