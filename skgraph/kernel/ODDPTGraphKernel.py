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
along with scikit-learn-graph.  If not, see <http://www.gnu.org/licenses/>.
"""


from graphKernel import GraphKernel
from graphKernel import normalize

#from scipy.sparse import csr_matrix
from ..graph.GraphTools import generateDAG
from ..graph.GraphTools import generateDAGOrdered
from ..graph.GraphTools import orderDAGvertices
import numpy as np
import sys
#from dependencies.pythontk import tree_kernels_new
from ..tree import tree_kernel_PT_new


from multiprocessing import Pool
from functools import partial

#suxiliary functions for parallelization
def calculate_kernel_dags(DAGS1,GraphKernel,DAGS2):
    return GraphKernel.kernelFunctionDAGS(DAGS1,DAGS2)
    
def calculate_kernel(Graph1,GraphKernel,Graph2):
    return GraphKernel.kernelFunction(Graph1,Graph2)


class ODDPTGraphKernel(GraphKernel):
    """
    Class that implements the ODDKernel with ST kernel
    """
    
    def __init__(self, r = 3, l = 1,m=1, normalization = True, hashsz=32):
        """
        Constructor
        @type r: integer number
        @param r: ODDKernel Parameter
        
        @type l: number in (0,1]
        @param l: ODDKernel Parameter
        
        @type normalization: boolean
        @param normalization: True to normalize the feature vectors
        
        @type hashsz: integer number
        @param hashsz: numbers of bits used to compute the hash function
        
        @type show: boolean
        @param show: If true shows graphs and DAGs during computation
        """
        #booleans indicating to cosider labels or not
        #beta is ignored for now ans hard-coded to 1/d
        self.Lambda=l
        self.mu=m
        self.max_radius=r
        self.normalization=normalization
        self.__hash_size=hashsz 
        self.__bitmask = pow(2, hashsz) - 1
        self.__startsymbol='!' #special symbols used in encoding
        self.__conjsymbol='#'
        self.__endsymbol='?'
        self.treekernelfunction=tree_kernel_PT_new.PTKernel(self.Lambda,self.mu,normalize=True)
        
    def kernelFunction(self,Graph1, Graph2):#TODO inutile
        #calculate the kernel between two graphs
        kernel=0
        #pass
        
        #treekernel=tree_kernel_PT_new.PTKernel(self.Lambda,self.mu,normalize=True)    

        #generate second DAGS
        DAGS2=[]
        for v2 in Graph2.nodes():
            #print Graph2.node[v2]['label']
            if Graph2.node[v2]['viewpoint']:
                if not Graph2.graph['ordered']:
                    (DAG,maxLevel)=generateDAG(Graph2, v2, self.max_radius)
                    orderDAGvertices(DAG)
                else:
                    (DAG,maxLevel)=generateDAGOrdered(Graph2, v2, self.max_radius)

                #print DAG.nodes()
                DAGS2.append(DAG)
        
        for v1 in Graph1.nodes():
            if Graph1.node[v1]['label'].endswith('-REL'):

                (DAG1,maxLevel)=generateDAGOrdered(Graph1, v1, self.max_radius)
                DAG1.graph['root']=v1
                for v2 in xrange(len(DAGS2)):
                    DAG2=DAGS2[v2] # node indices starts from 1
                    kernel += self.treekernelfunction.kernel(DAG1,DAG2)
        return kernel
    def kernelFunctionDAGS(self,DAGS1, DAGS2):#TODO inutile
        #calculate the kernel between two graphs
        kernel=0.0
        #pass

        
        for DAG1 in DAGS1:
            for DAG2 in DAGS2:
                    kernel += self.treekernelfunction.kernel(DAG1,DAG2)
        #print ",",
        return float(kernel)
        
        
    def computeKernelMatrixTrain(self,Graphs):
        print "Compute DAGs representation"
        DAGSX=[]
        for g in Graphs:
            DAGS=[]
            for v2 in g.nodes():
                #print Graph2.node[v2]['label']
                if g.node[v2]['viewpoint']:
                    if not g.graph['ordered']:
                        (DAG,maxLevel)=generateDAG(g, v2, self.max_radius)
                        orderDAGvertices(DAG)
                    else:
                        (DAG,maxLevel)=generateDAGOrdered(g, v2, self.max_radius)

                    #print DAG.nodes()
                    DAGS.append(DAG)
            DAGSX.append(DAGS)
        
        print "Computing gram matrix"
        Gram = np.empty(shape=(len(Graphs),len(Graphs)))
        progress=0
        for  i in xrange(0,len(Graphs)):
            for  j in xrange(i,len(Graphs)):
                #print i,j
                progress+=1
                Gram[i][j]=self.kernelFunctionDAGS(DAGSX[i],DAGSX[j])
                Gram[j][i]=Gram[i][j]
                if progress % 1000 ==0:
                    print "k",
                    sys.stdout.flush()
                elif progress % 100 ==0:
                    print ".",
                    sys.stdout.flush()
        if self.normalization:
            Gram = normalize(Gram)
        return Gram        
    def computeKernelMatrix(self,GraphsX,GraphsY):

        print "Compute DAGs representation"
        DAGSY=[]
        for g in GraphsY:
            DAGS=[]
            for v2 in g.nodes():
                #print Graph2.node[v2]['label']
                if g.node[v2]['viewpoint']:
                    if not g.graph['ordered']:
                        (DAG,maxLevel)=generateDAG(g, v2, self.max_radius)
                        orderDAGvertices(DAG)
                    else:
                        (DAG,maxLevel)=generateDAGOrdered(g, v2, self.max_radius)

                    #print DAG.nodes()
                    DAGS.append(DAG)
            DAGSY.append(DAGS)
        
        print "Computing gram matrix"
        Gram = np.empty(shape=(len(GraphsX),len(GraphsY)))
        progress=0
        for  i in xrange(0,len(GraphsX)):
            DAGS=[]
            for v2 in GraphsX[i].nodes():
                #print Graph2.node[v2]['label']
                if GraphsX[i].node[v2]['viewpoint']:
                    if not GraphsX[i].graph['ordered']:
                        (DAG,maxLevel)=generateDAG(GraphsX[i], v2, self.max_radius)
                        orderDAGvertices(DAG)
                    else:
                        (DAG,maxLevel)=generateDAGOrdered(GraphsX[i], v2, self.max_radius)

                    #print DAG.nodes()
                    DAGS.append(DAG)
            for  j in xrange(0,len(GraphsY)):
                #print i,j
                progress+=1
                Gram[i][j]=self.kernelFunctionDAGS(DAGS,DAGSY[j])
                #Gram[j][i]=Gram[i][j]
                if progress % 1000 ==0:
                    print "k",
                    sys.stdout.flush()
                elif progress % 100 ==0:
                    print ".",
                    sys.stdout.flush()
        if self.normalization:
            Gram = normalize(Gram)
        return Gram

    def computeKernelMatrixTrainParallel(self,Graphs,njobs):
        print "Compute DAGs representation"
        DAGSX=[]
        for g in Graphs:
            DAGS=[]
            for v2 in g.nodes():
                #print Graph2.node[v2]['label']
                if g.node[v2]['viewpoint']:
                    if not g.graph['ordered']:
                        (DAG,maxLevel)=generateDAG(g, v2, self.max_radius)
                        orderDAGvertices(DAG)
                    else:
                        (DAG,maxLevel)=generateDAGOrdered(g, v2, self.max_radius)

                    #print DAG.nodes()
                    DAGS.append(DAG)
            DAGSX.append(DAGS)
        
        print "Computing gram matrix"
        
        if njobs == -1:
            njobs = None

        pool = Pool(njobs)
        Gram = np.empty(shape=(len(Graphs),len(Graphs)))
        progress=0
        for  i in xrange(0,len(Graphs)):
            partial_calculate_kernel = partial(calculate_kernel_dags, GraphKernel=self,DAGS2=DAGSX[i])

                
            jth_column=pool.map(partial_calculate_kernel,[DAGSX[j] for j in xrange(i,len(Graphs))])
            for  j in xrange(i,len(Graphs)):
                #print i,j
                progress+=1
                Gram[i][j]=jth_column[j-i]
                Gram[j][i]=Gram[i][j]
                if progress % 1000 ==0:
                    print "k",
                    sys.stdout.flush()
                elif progress % 100 ==0:
                    print ".",
                    sys.stdout.flush()
        if self.normalization:
            Gram = normalize(Gram)
        return Gram    

    def computeKernelMatrixParallel(self,GraphsX,GraphsY,njobs):

        print "Compute DAGs representation"
        DAGSY=[]
        for g in GraphsY:
            DAGS=[]
            for v2 in g.nodes():
                #print Graph2.node[v2]['label']
                if g.node[v2]['viewpoint']:
                    if not g.graph['ordered']:
                        (DAG,maxLevel)=generateDAG(g, v2, self.max_radius)
                        orderDAGvertices(DAG)
                    else:
                        (DAG,maxLevel)=generateDAGOrdered(g, v2, self.max_radius)

                    #print DAG.nodes()
                    DAGS.append(DAG)
            DAGSY.append(DAGS)
        
        print "Computing gram matrix"
        
        if njobs == -1:
            njobs = None

        pool = Pool(njobs)
        Gram = np.empty(shape=(len(GraphsX),len(GraphsY)))
        progress=0
        for  i in xrange(0,len(GraphsX)):
            DAGS=[]
            for v2 in GraphsX[i].nodes():
                #print Graph2.node[v2]['label']
                if GraphsX[i].node[v2]['label'].endswith('-REL'):
                    #print v2
                    (DAG,maxLevel)=generateDAGOrdered(GraphsX[i], v2, self.max_radius)
                    DAG.graph['root']=v2
                    #print DAG.nodes()
                    DAGS.append(DAG)
            partial_calculate_kernel = partial(calculate_kernel_dags, GraphKernel=self,DAGS2=DAGS)

                
            jth_column=pool.map(partial_calculate_kernel,[DAGSY[j] for j in xrange(0,len(GraphsY))])
            for  j in xrange(0,len(GraphsY)):
                #print i,j
                progress+=1
                Gram[i][j]=jth_column[j]
                #Gram[j][i]=Gram[i][j]
                if progress % 1000 ==0:
                    print "k",
                    sys.stdout.flush()
                elif progress % 100 ==0:
                    print ".",
                    sys.stdout.flush()
        if self.normalization:
            Gram = normalize(Gram)
        return Gram   
  