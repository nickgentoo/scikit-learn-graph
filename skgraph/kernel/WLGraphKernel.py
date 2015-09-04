# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 12:04:44 2015

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

The code is from the following source.

Weisfeiler_Lehman graph kernel.

Python implementation of Nino Shervashidze Matlab code at:
http://mlcb.is.tuebingen.mpg.de/Mitarbeiter/Nino/Graphkernels/

Author : Sandro Vega Pons

License:
"""

import numpy as np
import networkx as nx
import copy
from KernelTools import convert_to_sparse_matrix
from graphKernel import GraphKernel
from scipy.sparse import dok_matrix

class WLGraphKernel(GraphKernel):
    """
    Weisfeiler_Lehman graph kernel.
    """
    def __init__(self, r = 1, normalization = False):
        self.h=r
        self.normalization=normalization
        self.__startsymbol='!' #special symbols used in encoding
        self.__conjsymbol='#'
        self.__endsymbol='?'
    
    def kernelFunction(self, g_1, g_2):
        """Compute the kernel value (similarity) between two graphs. 
        
        Parameters
        ----------
        g1 : networkx.Graph
            First graph.
        g2 : networkx.Graph
            Second graph.
        h : interger
            Number of iterations.
        nl : boolean
            Whether to use original node labels. True for using node labels
            saved in the attribute 'node_label'. False for using the node 
            degree of each node as node attribute.
        
        Returns
        -------        
        k : The similarity value between g1 and g2.
        """
        gl = [g_1, g_2]
        return self.computeGram(gl)[0, 1]

    def transform(self, graph_list):
        """
        TODO
        """
        n = len(graph_list) #number of graphs
        
        phi={} #dictionary representing the phi vector for each graph. phi[r][c]=v each row is a graph. each column is a feature
        #phi=dok_matrix()
        NodeIdToLabelId = [0] * n # NodeIdToLabelId[i][j] is labelid of node j in graph i
        label_lookup = {} #map from features to corresponding id
        label_counter = 0 #incremental value for label ids

        for i in xrange(n): #for each graph            
            NodeIdToLabelId[i] = {}

            for j in graph_list[i].nodes():
                enc="0"+self.__startsymbol+graph_list[i].node[j]['label']
                if enc not in label_lookup:#update label_lookup and label ids
                    label_lookup[enc] = label_counter
                    NodeIdToLabelId[i][j] = label_counter
                    label_counter += 1
                else:
                    NodeIdToLabelId[i][j] = label_lookup[enc]
                #print enc, label_lookup[enc]
                if (i,label_lookup[enc]) not in phi:
                    phi[i,label_lookup[enc]]=0
                phi[i,label_lookup[enc]]+=1
        
        ### MAIN LOOP
        it = 0
        NewNodeIdToLabelId = copy.deepcopy(NodeIdToLabelId)
        #label_lookup = {}

        while it < self.h:
            #label_lookup = {}

            for i in xrange(n): #for each graph
                for j in graph_list[i].nodes(): #for each node, consider its neighbourhood
                    neighbors=[]
                    for u in graph_list[i].neighbors(j):
                        #print u,
                        neighbors.append(NodeIdToLabelId[i][u])
                    neighbors.sort()
                    #print
                    long_label_string=str(it+1)+self.__startsymbol+str(NodeIdToLabelId[i][j])
                    for u in neighbors:
                        long_label_string+=self.__conjsymbol+str(u)
                    #long_label_string=long_label_string[:-1]+self.__endsymbol
                    if long_label_string not in label_lookup:
                        label_lookup[long_label_string] = label_counter
                        NewNodeIdToLabelId[i][j] = label_counter
                        label_counter += 1
                    else:
                        NewNodeIdToLabelId[i][j] = label_lookup[long_label_string]
                    #print long_label_string, NewNodeIdToLabelId[i][j]
    
                    if (i,NewNodeIdToLabelId[i][j]) not in phi:
                        phi[i,NewNodeIdToLabelId[i][j]]=0
                    phi[i,NewNodeIdToLabelId[i][j]]+=1
            
            NodeIdToLabelId = copy.deepcopy(NewNodeIdToLabelId)
            it = it + 1
        #return dok_matrix(phi.todense()).tocsr()
        return convert_to_sparse_matrix(phi)
#    def transform(self, graph_list):
#        """
#        TODO
#        """
#        n = len(graph_list) #number of graphs
#        
#        phi={} #dictionary representing the phi vector for each graph. phi[r][c]=v each row is a graph. each column is a feature
#        
#        NodeIdToLabelId = [0] * n # NodeIdToLabelId[i][j] is labelid of node j in graph i
#        label_lookup = {} #map from features to corresponding id
#        label_counter = 1 #incremental value for label ids
#        
#        for i in range(n): #for each graph            
#            NodeIdToLabelId[i] = {}
#
#            for j in graph_list[i].nodes():
#                #print graph_list[i].node[j]['label']
#                if not label_lookup.has_key("0|"+str(graph_list[i].node[j]['label'])):#update label_lookup and label ids
#                    label_lookup["0|"+str(graph_list[i].node[j]['label'])] = label_counter
#                    NodeIdToLabelId[i][j] = label_counter
#                    label_counter += 1
#                else:
#                    NodeIdToLabelId[i][j] = label_lookup["0|"+str(graph_list[i].node[j]['label'])]
#                
#                if not phi.has_key((i,label_lookup["0|"+str(graph_list[i].node[j]['label'])])):
#                    phi[(i,label_lookup["0|"+str(graph_list[i].node[j]['label'])])]=0
#                phi[(i,label_lookup["0|"+str(graph_list[i].node[j]['label'])])]+=1
#        
#        ### MAIN LOOP
#        it = 0
#        NewNodeIdToLabelId = copy.deepcopy(NodeIdToLabelId)
#        #NewNodeIdToLabelId =[0] * n 
#        while it < self.h:
#            label_lookup = {}
#
#            for i in range(n): #for each graph
#                for j in graph_list[i].nodes(): #for each node, consider its neighbourhood
#                    neighbors=[]
#                    for u in graph_list[i].neighbors(j):
#                        #print u
#                        neighbors.append(NodeIdToLabelId[i][u])
#                    neighbors.sort()
#                    if len(neighbors)==0:
#                        print "Empty neighbors"
#                    #MODIFICATO RISPETTO a TESSELLI str(it)+self.__startsymbol+
#                    long_label_string=str(it+1)+"|"+str(NodeIdToLabelId[i][j])+self.__startsymbol
#                    for u in neighbors:
#                        long_label_string+=str(u)+self.__conjsymbol
#                    #long_label_string=long_label_string[:-1]+self.__endsymbol
#                    long_label_string=long_label_string[:-1]+self.__endsymbol
#
#                    if len(neighbors)==0:
#                        print long_label_string
#
#                    if not label_lookup.has_key(long_label_string):
#                        label_lookup[long_label_string] = label_counter
#                        NewNodeIdToLabelId[i][j] = label_counter
#                        label_counter += 1
#                    else:
#                        NewNodeIdToLabelId[i][j] = label_lookup[long_label_string]
#                        
#                    if not phi.has_key((i,NewNodeIdToLabelId[i][j])):
#                        phi[(i,NewNodeIdToLabelId[i][j])]=0
#                    phi[(i,NewNodeIdToLabelId[i][j])]+=1
#            
#            NodeIdToLabelId = copy.deepcopy(NewNodeIdToLabelId)
#            it = it + 1
#        return convert_to_sparse_matrix(phi)
            
    def __normalization(self, gram):
        """
        TODO
        """
        if self.normalization:
            diagonal=np.diag(gram)
            a=np.tile(diagonal,(gram.shape[0],1))
            b=diagonal.reshape((gram.shape[0],1))
            b=np.tile(b,(1,gram.shape[1]))
            
            return gram/np.sqrt(a*b)
        else :
            return gram
    def computeKernelMatrixTrain(self,Graphs):
        return self.computeGram(Graphs)   
    def computeGram(self,g_it,precomputed=None):
        if precomputed is None:
            precomputed=self.transform(g_it)
        return self.__normalization(precomputed.dot(precomputed.T).todense()).tolist()


