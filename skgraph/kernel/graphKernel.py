# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:52:04 2015

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
along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
"""

class GraphKernel(object):
    def __init__():
        print "This method has to be instantiated in the derived class!"

    def transform(self, G_list, n_jobs = 1):
        print "This method has to be instantiated in the derived class!"
    
    def kernelFunction(self,Graph1, Graph2):
        print "This method has to be instantiated in the derived class!"

    def computeKernelMatrixTrain(self,Graphs):
        print "Computing gram matrix"
        Gram = np.empty(shape=(len(Graphs),len(Graphs)))
        progress=0
        for  i in xrange(0,len(Graphs)):
            for  j in xrange(i,len(Graphs)):
                #print i,j
                progress+=1
                Gram[i][j]=self.kernelFunction(Graphs[i],Graphs[j])
                Gram[j][i]=Gram[i][j]
                if progress % 1000 ==0:
                    print "k",
                    sys.stdout.flush()
                elif progress % 100 ==0:
                    print ".",
                    sys.stdout.flush()

        return Gram

    def computeKernelMatrixTrainParallel(self,Graphs,njobs=-1):
        #TODO finire
        print "Computing gram matrix"
        Gram = np.empty(shape=(len(Graphs),len(Graphs)))
        progress=0
        
        if njobs == -1:
            njobs = None

        pool = Pool(njobs)
        

        for  i in xrange(0,len(Graphs)):
            partial_calculate_kernel = partial(calculate_kernel, GraphKernel=self,Graph2=Graphs[i])

                
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

def normalize(Gram):
    """
    Function that normalizes a squared gram matrix.
    """
    import copy
    from math import sqrt
    Gram_norm = copy.copy(Gram)
    for i in xrange(Gram.shape[0]):
        for j in xrange(Gram.shape[1]):
            Gram_norm[i][j] = Gram[i][j]/sqrt(Gram[i][i]*Gram[j][j])
    return Gram_norm