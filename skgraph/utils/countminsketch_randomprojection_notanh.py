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
#import array
import numpy as np
from numpy import median
import numpy.matlib
import copy
from itertools import izip
from numpy import random, sqrt, log, sin, cos, pi
from scipy.sparse import csr_matrix

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
        #self.tables = numpy.matlib.zeros(shape=(m,samplesize))
        #self.tables=numpy.random.normal(size=(m,samplesize))
#        for _ in xrange(d):
#            table = array.array("d", (0.0 for _ in xrange(m)))
#            self.tables.append(table)

    def transform(self, vector):
        #print "example size", vector.shape
        #print "transformation size", self.tables.shape
        #tables=csr_matrix ((self.m,self.samplesize))

        indices=vector.nonzero()[0]
        row=[]
        col=[]
        data=[]
        #print indices
        for i in indices:
            numpy.random.seed(i+(self.rs*10000))
            v=numpy.random.normal(0,1,self.m)
            v=numpy.multiply(sqrt(self.m),v)
            row.extend([idx for idx in xrange(self.m)])
            col.extend([i for idx in xrange(self.m)])
            data.extend(v)
        tables=csr_matrix ((data,(row,col)), shape=(self.m,self.samplesize))
        transformation= (tables*vector).todense()
        #print transformation.shape
        return transformation


