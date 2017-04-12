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
def next_greater_power_of_2(x):
    return 2**(x-1).bit_length()

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
    def tablesasarray(self):
        #print  np.asarray(self.tables).reshape(-1)
        return  np.asarray(self.tables).reshape(-1)
    def __init__(self, m, d):
        """ `m` is the size of the hash tables, larger implies smaller
        overestimation. `d` the amount of hash tables, larger implies lower
        probability of overestimation.
        """
        if not m or not d:
            raise ValueError("Table size (m) and amount of hash functions (d)"
                             " must be non-zero")
        self.m = m
        self.d = d
        self.n = 0
        #self.hashsize1=next_greater_power_of_2(self.d)
        #self.hashsize2=next_greater_power_of_2(self.m)

        self.hash1 = numpy.matlib.zeros(shape=(d,1))
        self.tables = numpy.matlib.zeros(shape=(d,m))
#        for _ in xrange(d):
#            table = array.array("d", (0.0 for _ in xrange(m)))
#            self.tables.append(table)

    def _hash1(self,x):
        md5 = hashlib.md5(str(hash(x))+"-1")
        return int(md5.hexdigest(), 16) % self.d

    def _hash2(self,x):
        md5 = hashlib.md5(str(hash(x))+"2")
        return int(md5.hexdigest(), 16) % self.m

    def _hash(self, x):
        md5 = hashlib.md5(str(hash(x)))
        for i in xrange(self.d):
            md5.update(str(i))
            yield int(md5.hexdigest(), self.hashsize2) % self.m

    def add(self, x, value=1.0):
        """
        Count element `x` as if had appeared `value` times.
        By default `value=1` so:

            sketch.add(x)

        Effectively counts `x` as occurring once.
        """
        self.n += value
        index=self._hash1(x)
        self.hash1[index]+= value
        self.tables[index,self._hash2(x)] += value

    def add_from_vector(self, V):
        """
        Count element `x` as if had appeared `value` times.
        By default `value=1` so:

            sketch.add(x)

        Effectively counts `x` as occurring once.
        """
        for x,value in zip(xrange(len(V)),V):
            self.n += value
            index = self._hash1(x)
            self.hash1[index] += value
            assert isinstance(value, float)
            self.tables[index, self.hash2(x)] += value

    def query(self, x):
        """
        Return an estimation of the amount of times `x` has ocurred.
        The returned value always overestimates the real value.
        """
        #Modified by Nicolo' Navarin
        return median([self.tables[tableIndex,i] for tableIndex, i in izip(xrange(self.d), self._hash(x))])
        #return min(table[i] for table, i in izip(self.tables, self._hash(x)))

    def __getitem__(self, x):
        """
        A convenience method to call `query`.
        """
        return self.query(x)

    def __len__(self):
        """
        The amount of things counted. Takes into account that the `value`
        argument of `add` might be different from 1.
        """
        return self.n
#==============================================================================
#     def dot(self, other):
#         dots=[]
#         for i in xrange(self.d):
#             temp=self.tables[i,:]*other.tables[i,:].T
#             dots.append(temp)
#         #return median(dots) median is not a kernel!
#         return np.average(dots)
#==============================================================================
    def dot(self, other):
        #this version implements the average
        #sumfirst=self.hash1.dot(other.hash1)
        sumfirst=np.asarray(self.hash1).reshape(-1).dot(np.asarray(other.hash1).reshape(-1))
        return ((self.tablesasarray().dot(other.tablesasarray())) + sumfirst) /2.0
        #return median(dots) median is not a kernel!
        
    def __add__(self, other):
        temp=copy.deepcopy(self)
        temp.tables+=other.tables
        temp.hash1+=other.hash1
#        for i in xrange(self.d):
#            for j in xrange(self.m):
#                temp.tables[i][j]+=other.tables[i][j]
        return temp
    def __iadd__(self, other):
        temp=self
        temp.tables+=other.tables
        temp.hash1+=other.hash1

#        for i in xrange(self.d):
#            for j in xrange(self.m):
#                temp.tables[i][j]+=other.tables[i][j]
        return temp
    
    def __sub__(self, other):
        temp=copy.deepcopy(self)
        temp.tables-=other.tables
        temp.hash1-=other.hash1


#        for i in xrange(self.d):
#            for j in xrange(self.m):
#                temp.tables[i][j]-=other.tables[i][j]
        return temp
    def __isub__(self, other):
        temp=self
        temp.tables-=other.tables
        temp.hash1-=other.hash1


#        for i in xrange(self.d):
#            for j in xrange(self.m):
#                temp.tables[i][j]-=other.tables[i][j]
        return temp
    
    def __mul__(self, number):
        temp=copy.deepcopy(self)
        temp.tables=temp.tables*number
        temp.hash1=temp.hash1*number


#==============================================================================
#         for i in xrange(self.d):
#             for j in xrange(self.m):
#                 temp.tables[i][j]= temp.tables[i][j]*number
#                 #print temp.tables[i][j]
#==============================================================================
        return temp
    def __imul__(self, number):
        temp=self
        temp.tables=temp.tables*number
        temp.hash1=temp.hash1*number


#==============================================================================
#         for i in xrange(self.d):
#             for j in xrange(self.m):
#                 temp.tables[i][j]= temp.tables[i][j]*number
#                 #print temp.tables[i][j]
#==============================================================================
        return temp
        
    __rmul__ = __mul__
    __radd__ = __add__
