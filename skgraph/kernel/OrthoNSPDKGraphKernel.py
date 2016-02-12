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
import copy
from KernelTools import convert_to_sparse_matrix
from graphKernel import GraphKernel
from skgraph.feature_extraction.graph.NSPDK.NSPDKVectorizer import NSPDKVectorizer
class NSPDKGraphKernel(GraphKernel):
    """
    Weisfeiler_Lehman graph kernel.
    """
    def __init__(self, r = 1, d=1, normalization = False):
        self.h=r
        self.d=d
        self.normalization=normalization
        self.vectorizer=NSPDKVectorizer(r=self.h,d=self.d,normalization=self.normalization)

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
        return self.vectorizer.transform(graph_list)
            
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
        return self.computeGrams(Graphs)   

    def computeGrams(self, g_it, precomputed =None):
        if precomputed is None:
            precomputed = self.transform(g_it)

        return [np.array(self.__normalization(p.dot(p.T).todense())) for p in precomputed]

