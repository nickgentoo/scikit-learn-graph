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
"""#from .... import graph
#from skgraph import graph
import numpy as np
import sys
import networkx as nx
from ..kernel.vector_kernels import gaussianKernel




class STCLKernel:
    """
    SubTree kernel for trees with Continuous Labels
    """
    def __init__(self, l, hashsep="#", savemem=False):
        self.l = float(l)
        self.hashsep = hashsep
        self.cache = {}
        self.savemem = savemem
#FeatureVector ={subtreeID:[[freq,size,VecLabels] [freq,size,VecLabels] ] }
    def generateTreeFeatureMap(self, Tree):
        """
        Generate the STCL FeatureMap representation from a tree.
        """
        FeatureMap = {}

        self._addToFeatureMap(FeatureMap, Tree)
        return FeatureMap


    def _addToFeatureMap(self, FeatureVector, D2):
        """
        Adds the STCL FeatureMap of D2 to FeatureVector.
        """
        #setOrder(D2,D2.graph['root'])
        if not D2.graph['ordered']:
            print "ERROR: DAG is not ordered!"

        for u in nx.topological_sort(D2)[::-1]:
            subtree_str, size = setHashSubtreeIdentifierSize(D2, u)
            #print subtree_str
            if not subtree_str in FeatureVector:
                #FeatureVector[subtree_str]=[[1,size,D2.node[u]['veclabel']]]
                FeatureVector[subtree_str] = {tuple(D2.node[u]['veclabel']):[1,size]}
            else:
                #c'e il subtree, vediamo se c'e l'etichetta vettoriale
                VecLabels = D2.node[u]['veclabel']

                if tuple(VecLabels) in FeatureVector[subtree_str]:
                    FeatureVector[subtree_str][tuple(VecLabels)][0] += 1
                else:
                    FeatureVector[subtree_str][tuple(VecLabels)] = [1, size]
    def calculateNodekernel(self, v1, v2, beta):
        """
        Calculates the kernel between two labels (vectors).
        """
        if str(v1)+str(v2) in self.cache:
            return self.cache[str(v1)+str(v2)]
        if str(v2)+str(v1) in self.cache:
            return self.cache[str(v2)+str(v1)]

        labels_kernel = gaussianKernel(v1, v2, beta)
        self.cache[str(v1)+str(v2)] = labels_kernel
        return labels_kernel

    def kernelFeatureMap(self, FeatureVector1, FeatureVector2):
        """
        Calculates the kernel value between two STCL FeatureMaps.
        """
        #Assumes ha and hb are ordered list of pairs (subtreeid, subtreesize)
        #a.kernelreprst,b.kernelreprst are checked or created in preProcess()
        if self.savemem:
            self.cache = {}
        k = 0
        for key in FeatureVector1:
            if key in FeatureVector2:
                for v1 in FeatureVector1[key]:
                    for v2 in FeatureVector2[key]:
                        labels_kernel = self.calculateNodekernel(v1, v2, 1.0/len(v1))
                        f1, s1 = FeatureVector1[key][v1]
                        f2, s2 = FeatureVector2[key][v2]

                        k += f1*f2*labels_kernel*self.l**s1
        return k

    def kernel(self, Tree1, Tree2):
        """
        Calculates the STCL kernel between two trees.
        Assumes the trees are ordered.
        """
        #Assumes ha and hb are ordered list of pairs (subtreeid, subtreesize)
        #a.kernelreprst,b.kernelreprst are checked or created in preProcess()
        assert(Tree1.graph['ordered'] and Tree2.graph['ordered']), 'ERROR: trees are not ordered'        
        if self.savemem:
            self.cache = {}
        FeatureMap1 = self.generateTreeFeatureMap(Tree1)
        FeatureMap2 = self.generateTreeFeatureMap(Tree2)
        return  self.kernelFeatureMap(FeatureMap1, FeatureMap2)

    def __str__(self):
        """
        Generate a string with kernel information.
        """
        return "Subtree Kernel, with lambda=" + self.l

    #requires feature vectors
    def computeKernelMatrix(self, trees):
        """
        Computes the Gram matrix from examples in trees.
        @param trees: a list of networkx graphs that are trees.
        """
        print "Computing gram matrix"
        Gram = np.empty(shape=(len(trees), len(trees)))
        progress = 0
        for  i in xrange(0, len(trees)):
            for  j in xrange(i, len(trees)):
                #print "COMPUTING GRAPHS",i,j
                progress += 1
                Gram[i][j] = self.kernel(trees[i], trees[j])
                Gram[j][i] = Gram[i][j]
                if progress % 1000 == 0:
                    print "k",
                    sys.stdout.flush()
                elif progress % 100 == 0:
                    print ".",
                    sys.stdout.flush()
        return Gram

## GRAPH FUNCTIONS

def setHashSubtreeIdentifierSize(T, nodeID, sep='|'):
    """
    The method computes an identifier of the node based on
    1) the label of the node self
    2) the hash values of the children of self
    For each visited node the hash value is stored into the attribute subtreeId
    The label and the identifiers of the children nodes are separated by the char 'sep'
    """
    #assume ordered children
    if 'subtreeIDST' in T.node[nodeID]:
        return T.node[nodeID]['subtreeIDST'], T.node[nodeID]['subtreeSize']
    stri = str(T.node[nodeID]['label'])
    size = 1
    #print stri
    if stri.find(sep) != -1:
        print "ERROR: identifier " + sep + "used in label. Please set it with setHashSep(newsep)"
    for c in T.node[nodeID]['childrenOrder']:#T.successors(nodeID):
        #print "children exists"
        child_str, child_size = setHashSubtreeIdentifierSize(T, c, sep)
        stri += sep + child_str
        size += child_size
    #print stri
    T.node[nodeID]['subtreeSize'] = size
    T.node[nodeID]['subtreeIDST'] = str(hash(stri)) #hash()
    #print T.node[nodeID]['subtreeIDST'],T.node[nodeID]['subtreeSize']
    return T.node[nodeID]['subtreeIDST'], T.node[nodeID]['subtreeSize']
