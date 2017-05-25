"""
Weisfeiler_Lehman graph kernel.

Python implementation of Nino Shervashidze Matlab code at:
http://mlcb.is.tuebingen.mpg.de/Mitarbeiter/Nino/Graphkernels/

Author : Sandro Vega Pons

License:
"""

import numpy as np
import networkx as nx
import copy
import math
from KernelTools import convert_to_sparse_matrix
from graphKernel import GraphKernel
from sklearn import preprocessing as pp

class WLCGraphKernel(GraphKernel):
    """
    Weisfeiler_Lehman graph kernel.
    """
    def __init__(self, r =1, normalization =True, version =1, show =False):
        self.h=r
        self.normalization=normalization
        self.show=show #TODO implementa
        self.__startsymbol='!' #special symbols used in encoding
        self.__conjsymbol='#'
        self.__endsymbol='?'
        self.__contextsymbol='@'
        self.__fsfeatsymbol='*'
        self.__version=version
    
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
        
        NodeIdToLabelId = [0] * n # NodeIdToLabelId[i][j] is labelid of node j in graph i
        label_lookup = {} #map from features to corresponding id
        label_counter = 0 #incremental value for label ids

        for i in range(n): #for each graph            
            NodeIdToLabelId[i] = {}

            for j in graph_list[i].nodes(): #for each node
                if not label_lookup.has_key(graph_list[i].node[j]['label']):#update label_lookup and label ids from first iteration that consider node's labels
                    label_lookup[graph_list[i].node[j]['label']] = label_counter
                    NodeIdToLabelId[i][j] = label_counter
                    label_counter += 1
                else:
                    NodeIdToLabelId[i][j] = label_lookup[graph_list[i].node[j]['label']]
                
                if self.__version==0: #consider old FS features
                    feature=self.__fsfeatsymbol+str(label_lookup[graph_list[i].node[j]['label']])
                    if not phi.has_key((i,feature)):
                        phi[(i,feature)]=0.0
                    phi[(i,feature)]+=1.0
        
        ### MAIN LOOP
        it = 0
        NewNodeIdToLabelId = copy.deepcopy(NodeIdToLabelId) #labels id of nex iteration
        
        while it <= self.h: #each iteration compute the next labellings (that are contexts of the previous)
            label_lookup = {}

            for i in range(n): #for each graph
                for j in graph_list[i].nodes(): #for each node, consider its neighbourhood
                    neighbors=[]
                    for u in graph_list[i].neighbors(j):
                        neighbors.append(NodeIdToLabelId[i][u])
                    neighbors.sort() #sorting neighbours
                    
                    long_label_string=str(NodeIdToLabelId[i][j])+self.__startsymbol #compute new labels id
                    for u in neighbors:
                        long_label_string+=str(u)+self.__conjsymbol
                    long_label_string=long_label_string[:-1]+self.__endsymbol
                    
                    if not label_lookup.has_key(long_label_string):
                        label_lookup[long_label_string] = label_counter
                        NewNodeIdToLabelId[i][j] = label_counter
                        label_counter += 1
                    else:
                        NewNodeIdToLabelId[i][j] = label_lookup[long_label_string]
                        
                    if self.__version==0 and it<self.h: #consider FS features
                        feature=self.__fsfeatsymbol+str(NewNodeIdToLabelId[i][j])
                        if not phi.has_key((i,feature)):
                            phi[(i,feature)]=0.0
                        phi[(i,feature)]+=1.0
                    
                    #adding feature with contexts
                    if it<self.h:
                        feature=str(NodeIdToLabelId[i][j])+self.__contextsymbol+str(NewNodeIdToLabelId[i][j]) #with context
                    else:
                        feature=str(NodeIdToLabelId[i][j]) #null context
                    if not phi.has_key((i,feature)):
                        phi[(i,feature)]=0.0
                    phi[(i,feature)]+=1.0
            
            NodeIdToLabelId = copy.deepcopy(NewNodeIdToLabelId) #update current labels id
            it = it + 1
        
        ve=convert_to_sparse_matrix(phi)    
        if self.normalization:
             ve = pp.normalize(ve, norm='l2', axis=1)
        return ve
#        return self.__normalization(phi)
            
#    def __normalization(self, feature_list):
#        """
#        Private method that normalize the feature vector if requested
#        @type feature_list: Dictionary
#        @param feature_list: Dictionary that represent the feature vector
#        
#        @rtype: Dictionary
#        @return: The normalized feature vector
#        """
#        if self.normalization:
#            total_norm = 0.0
#        
#            for value in feature_list.itervalues():
#                total_norm += value*value
#            
#            normalized_feature_vector = {}
#            sqrt_total_norm = math.sqrt( float(total_norm) )
#            for (key,value) in feature_list.iteritems():
#                normalized_feature_vector[key] = value/sqrt_total_norm
#            return normalized_feature_vector
#        else :
#            return dict(feature_list)

    def computeGram(self,g_it,precomputed=None):
        if precomputed is None:
            precomputed=self.transform(g_it)
        return precomputed.dot(precomputed.T).todense().tolist()

    def computeKernelMatrixTrain(self,Graphs):
        return self.computeGram(Graphs)
    
# if __name__=='__main__': #TODO converti in test
#     #g=nx.Graph()
#     """
#     g.add_node(1,label='F')
#     g.add_node(2,label='B')
#     g.add_node(3,label='F')
#     g.add_node(4,label='A')
#     g.add_node(5,label='C')
#     g.add_node(6,label='B')
#     g.add_node(7,label='E')
#     g.add_node(8,label='H')
#     g.add_node(9,label='G')
#     g.add_node(10,label='I')
#     g.add_edge(1, 2)
#     g.add_edge(1, 3)
#     g.add_edge(1, 6)
#     g.add_edge(1, 7)
#     g.add_edge(2, 4)
#     g.add_edge(3, 4)
#     g.add_edge(4, 5)
#     g.add_edge(7, 8)
#     g.add_edge(8, 9)
#     g.add_edge(9, 10)
#     g.add_edge(6, 4)
#     """
#     """
#     g1=nx.Graph()
#     g1.add_node(0,label='A')
#     g1.add_node(1,label='B')
#     g1.add_node(2,label='C')
#     g1.add_node(3,label='D')
#     #g.add_node(5,label='A')
#     g1.add_edge(0, 1)
#     g1.add_edge(1, 2)
#     g1.add_edge(0, 3)
#     #g.add_edge(3, 4)
#     #g.add_edge(4, 5)
#     """
#     
#     g1=nx.Graph()
#     g1.add_node(0,label='E')
#     g1.add_node(1,label='B')
#     g1.add_node(2,label='D')
#     g1.add_node(3,label='C')
#     g1.add_node(4,label='A')
#     g1.add_node(5,label='A')
#     g1.add_edge(0,1)
#     g1.add_edge(0,2)
#     g1.add_edge(0,3)
#     g1.add_edge(1,3)
#     g1.add_edge(2,3)
#     g1.add_edge(2,4)
#     g1.add_edge(2,5)
#     
#     g2=nx.Graph()
#     g2.add_node(0,label='B')
#     g2.add_node(1,label='E')
#     g2.add_node(2,label='D')
#     g2.add_node(3,label='C')
#     g2.add_node(4,label='A')
#     g2.add_node(5,label='B')
#     g2.add_edge(0,1)
#     g2.add_edge(0,2)
#     g2.add_edge(1,2)
#     g2.add_edge(1,3)
#     g2.add_edge(2,3)
#     g2.add_edge(2,4)
#     g2.add_edge(3,5)
#     
#     ke=FSContextsKernel(h=1, normalization = False,version=1,show=False)
#     print ke.computeGram([g1,g2])
#     print ke.kernelFunction(g1, g2)
