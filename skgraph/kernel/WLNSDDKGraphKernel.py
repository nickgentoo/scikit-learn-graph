__author__ = "Carlo Maria Massimo"
__date__ = "07/oct/2015"
__credits__ = ["Carlo Maria Massimo"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer = "Carlo Maria Massimo"
__email__ = "cmassim@gmail.com"
__status__ = "Development"

import numpy as np
import networkx as nx
import copy
import math
import sys
from sklearn import preprocessing as pp
from KernelTools import convert_to_sparse_matrix
from graphKernel import GraphKernel
from scipy.sparse import dok_matrix
from skgraph.kernel.ODDSTGraphKernel import ODDSTGraphKernel
from ..graph.GraphTools import drawGraph
from ..graph.GraphTools import generateDAG
from ..graph.GraphTools import generateDAGOrdered
from ..graph.GraphTools import orderDAGvertices

class WLNSDDKGraphKernel(GraphKernel):
    """
    Fast Subtree kernel with ODDK base kernel.
    """
    def __init__(self, k, h = 1, l = 1, normalization = False):
        # depth of "neighborhood" considered
        self.k = k
        self.Lambda = l

        # number of iterations of the WL test
        self.h = h
        self.normalization = normalization

        #special symbols used in encoding
        self.__startsymbol = '!'
        self.__conjsymbol = '#'
        self.__endsymbol = '?'
        self.__fsfeatsymbol = '*'

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

        for i in range(n): #for each graph            
            r = 1

            while (r < self.k+1):
                phi.update({(i, k): v for (k, v) in self.getFeaturesApproximated(graph_list[i], r, self.h).items()})
                r += 1
                    
        ve=convert_to_sparse_matrix(phi)    
        if self.normalization:
             ve = pp.normalize(ve, norm='l2', axis=1)
        return ve

    def getFeaturesApproximated(self, G, radius, iterations):
        """
        Public method that given a networkx graph G will create the dictionary representing its features according to the ST Kernel.
        The computation will use a hash function to encode a feature. There might be collisions
        @type G: networkx graph
        @param G: the graph to extract features from
        
        @type hashsize: integer
        @param hashsize: number of bits of the hash function to use
        
        @type MapNodeToNextLabel: self.UniqueMap
        @param MapNodeToNextLabel: Map between feature's encodings and integer values
        
        @rtype: dictionary
        @return: the encoding-feature dictionary
        """
        Dict_features={}

        next_iteration_G = G.copy()

        for v in G.nodes():
            if G.node[v]['viewpoint']:
                if not G.graph['ordered']:
                    (DAG, maxLevel) = generateDAG(G, v, radius)
                    #orderDAGvertices(DAG)
                else:
                    (DAG, maxLevel) = generateDAGOrdered(G, v, radius)
            
                    
                MapNodeToProductionsID={} #k:list(unsigned)
                MapNodetoFrequencies={} #k:list(int)
                for u in DAG.nodes():
                    MapNodeToProductionsID[u]=[]
                    MapNodetoFrequencies[u]=[]
                MapProductionIDtoSize={} #k:int
        
                reverse_toposort = nx.topological_sort(DAG)[::-1]
                for u in reverse_toposort:
                    max_child_height=0
                    for child in DAG.successors(u):
                        child_height=len(MapNodeToProductionsID.get(child))
                        
                        if child_height > max_child_height:
                            max_child_height = child_height
                            
                    for depth in xrange(max_child_height+1):
                        if depth==0:
                            enc=hash(str(DAG.node[u]['label']))
                            
                            MapNodeToProductionsID[u].append(enc)
                            
                            frequency=0
                            if max_child_height==0:
                                frequency=maxLevel - DAG.node[u]['depth']
                            
                            MapNodetoFrequencies[u].append(frequency)
                            MapProductionIDtoSize[enc]=1

                            # if u is the last node from the reversed toposort
                            # add the feature to the feature dictionary,
                            # rename it for the next iteration graph
                            if u == reverse_toposort[-1]:
                                if Dict_features.get(enc) is None:
                                    Dict_features[enc]=float(frequency+1.0)*math.sqrt(self.Lambda)
                                else:
                                    Dict_features[enc]+=float(frequency+1.0)*math.sqrt(self.Lambda)
                            
                                next_iteration_G.node[u]['label'] = enc
                            
                        else:
                            size=0
                            encoding=str(DAG.node[u]['label'])
                            
                            vertex_label_id_list=[]#list[string]
                            min_freq_children=sys.maxint
                            
                            for child in DAG.successors(u):
                                size_map=len(MapNodeToProductionsID[child])
                                child_hash=MapNodeToProductionsID[child][min(size_map,depth)-1]
                                freq_child=MapNodetoFrequencies[child][min(size_map,depth)-1]
                                
                                if freq_child<min_freq_children:
                                    min_freq_children=freq_child
                                
                                vertex_label_id_list.append(child_hash)
                                size+=MapProductionIDtoSize[child_hash]
                            
                            vertex_label_id_list.sort()
                            encoding+=self.__startsymbol+str(vertex_label_id_list[0])
                            
                            for i in range(1,len(vertex_label_id_list)):
                                encoding+=self.__conjsymbol+str(vertex_label_id_list[i])
                            
                            encoding+=self.__endsymbol
                            encoding=hash(encoding)
                            
                            MapNodeToProductionsID[u].append(encoding)
                            size+=1
                            MapProductionIDtoSize[encoding]=size
                            
                            frequency = min_freq_children
                            MapNodetoFrequencies[u].append(frequency)
                            
                            # if u is the last node from the reversed toposort
                            # add the feature to the feature dictionary,
                            # rename it for the next iteration graph
                            if u == reverse_toposort[-1]:
                                if Dict_features.get(encoding) is None:
                                    Dict_features[encoding]=float(frequency+1.0)*math.sqrt(math.pow(self.Lambda,size))
                                else:
                                    Dict_features[encoding]+=float(frequency+1.0)*math.sqrt(math.pow(self.Lambda,size))
                                next_iteration_G.node[u]['label'] = encoding
                            
        if (iterations > 0):
            # recursive call with iterations-1 and relabeled graph
            # join results with Dict_features
            Dict_features.update(self.getFeaturesApproximated(next_iteration_G, radius, iterations-1))

        return Dict_features

    def computeKernelMatrixTrain(self,Graphs):
        return self.computeGram(Graphs)   

    def computeGram(self,g_it,precomputed=None):
        if precomputed is None:
            precomputed=self.transform(g_it)
        return precomputed.dot(precomputed.T).todense().tolist()


