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
from numpy import median, average
import numpy.matlib
import numpy as np
import copy
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
    def asarray(self):
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
        self.tables = numpy.matlib.zeros(shape=(d,m))
#        for _ in xrange(d):
#            table = array.array("d", (0.0 for _ in xrange(m)))
#            self.tables.append(table)

    def _hash(self, x):
        md5 = hashlib.md5(str(hash(x)))
        for i in xrange(self.d):
            md5.update(str(i))
            yield int(md5.hexdigest(), 16) % self.m
    def _sign_hash(self, x):
                md5sign = hashlib.md5(str(hash(x)))
                for i in xrange(self.d):
                    md5sign.update(str(i+42))
                    boolsign= int(md5sign.hexdigest(), 16) % 2
                    boolsign=1
                    if boolsign:
                        y=-1
                    yield boolsign

    def add(self, x, value=1.0):
        """
        Count element `x` as if had appeared `value` times.
        By default `value=1` so:

            sketch.add(x)

        Effectively counts `x` as occurring once.
        """
        self.n += value
        for tableIndex, i, sign in zip(xrange(self.d), self._hash(x), self._sign_hash(x)):
            self.tables[tableIndex,i] += value*sign

    def query(self, x):
        """
        Return an estimation of the amount of times `x` has ocurred.
        The returned value always overestimates the real value.
        """
        #Modified by Nicolo' Navarin
        return average([self.tables[tableIndex,i] for tableIndex, i in zip(xrange(self.d), self._hash(x))])
        #return min(table[i] for table, i in zip(self.tables, self._hash(x)))

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
    def dot(self, other):
        return (self.asarray().dot(other.asarray()))/ float(self.d)

    def __add__(self, other):
        temp=copy.deepcopy(self)
        temp.tables+=other.tables
#        for i in xrange(self.d):
#            for j in xrange(self.m):
#                temp.tables[i][j]+=other.tables[i][j]
        return temp
    def __iadd__(self, other):
        temp=self
        temp.tables+=other.tables
#        for i in xrange(self.d):
#            for j in xrange(self.m):
#                temp.tables[i][j]+=other.tables[i][j]
        return temp
    
    def __sub__(self, other):
        temp=copy.deepcopy(self)
        temp.tables-=other.tables

#        for i in xrange(self.d):
#            for j in xrange(self.m):
#                temp.tables[i][j]-=other.tables[i][j]
        return temp
    def __isub__(self, other):
        temp=self
        temp.tables-=other.tables

#        for i in xrange(self.d):
#            for j in xrange(self.m):
#                temp.tables[i][j]-=other.tables[i][j]
        return temp
    
    def __mul__(self, number):
        temp=copy.deepcopy(self)
        temp.tables=temp.tables*number

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

#==============================================================================
#         for i in xrange(self.d):
#             for j in xrange(self.m):
#                 temp.tables[i][j]= temp.tables[i][j]*number
#                 #print temp.tables[i][j]
#==============================================================================
        return temp
        
    __rmul__ = __mul__
    __radd__ = __add__
