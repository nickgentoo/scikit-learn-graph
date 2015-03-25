# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:26:01 2015

Copyright 2015 Nicolo' Navarin

This file is part of scikit-learn-graph.

scikit-learn-graph is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

scikit-learn-graph is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with scikit-learn-graph.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import sys
from multiprocessing import Pool
from functools import partial
def calculate_kernel(Graph1,TreeKernel,Graph2):
    return TreeKernel.kernel(Graph1,Graph2)
class TreeKernel:
    def __init__():
        print 'To be defined in subclasses'
    
    def kernel(self, a, b):
        """compute the tree kernel on the trees a and b"""
        print 'To be defined in subclasses'
    
    def computeKernelMatrixTrain(self,Graphs):
        #TODO
        print "Computing gram matrix"
        Gram = np.empty(shape=(len(Graphs),len(Graphs)))
        progress=0
        for  i in xrange(0,len(Graphs)):
            for  j in xrange(i,len(Graphs)):
                #print "COMPUTING GRAPHS",i,j
                progress+=1
                Gram[i][j]=self.kernel(Graphs[i],Graphs[j])
                Gram[j][i]=Gram[i][j]
                if progress % 1000 ==0:
                    print "k",
                    sys.stdout.flush()
                elif progress % 100 ==0:
                    print ".",
                    sys.stdout.flush()
    
        return Gram
        
        def computeKernelMatrix(self,Graphs1, Graphs2):
            #TODO implementare
            print "Computing gram matrix"
    #        Gram = np.empty(shape=(len(Graphs),len(Graphs)))
    #        progress=0
    #        for  i in xrange(0,len(Graphs)):
    #            for  j in xrange(i,len(Graphs)):
    #                #print "COMPUTING GRAPHS",i,j
    #                progress+=1
    #                Gram[i][j]=self.kernel(Graphs[i],Graphs[j])
    #                Gram[j][i]=Gram[i][j]
    #                if progress % 1000 ==0:
    #                    print "k",
    #                    sys.stdout.flush()
    #                elif progress % 100 ==0:
    #                    print ".",
    #                    sys.stdout.flush()
    #    
    #        return Gram
    
    def computeKernelMatrixTrainParallel(self,Graphs,njobs=-1):
        """
        Compute a symmetric Gram matrix in prarallel on njobs cores        
        """
        print "Computing gram matrix"
        Gram = np.empty(shape=(len(Graphs),len(Graphs)))
        progress=0
        
        if njobs == -1:
            njobs = None

        pool = Pool(njobs)
        

        for  i in xrange(0,len(Graphs)):
            partial_calculate_kernel = partial(calculate_kernel, TreeKernel=self,Graph2=Graphs[i])

                
            jth_column=pool.map(partial_calculate_kernel,[Graphs[j] for j in range(i,len(Graphs))])
            for  j in xrange(i,len(Graphs)):
                #print i,j-i

                progress+=1
                Gram[i][j]=jth_column[j-i]
                Gram[j][i]=Gram[i][j]
                if progress % 1000 ==0:
                    print "k",
                    sys.stdout.flush()
                elif progress % 100 ==0:
                    print ".",
                    sys.stdout.flush()

        return Gram
