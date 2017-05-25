# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:52:04 2015

Copyright 2015 Nicolo' Navarin, Riccardo Tesselli

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

from skgraph.kernel.graphKernel import GraphKernel
from scipy.sparse import csr_matrix
from ..graph.GraphTools import generateDAG
from ..graph.GraphTools import generateDAGOrdered
from ..graph.GraphTools import orderDAGvertices

import networkx as nx
import numpy as np
import math
from collections import defaultdict
import sys

class ODDSTOrthogonalizedGraphKernel(GraphKernel):
    """
    Class that implements the ODDKernel with ST kernel
    """
    def __init__(self, r=3, l=1, normalization=True, hashsz=31):
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
        self.Lambda = l
        self.max_radius = r
        self.normalization = normalization
        self.__hash_size = hashsz 
        self.__bitmask = pow(2, hashsz) - 1
        self.__startsymbol = '!' #special symbols used in encoding
        self.__conjsymbol = '#'
        self.__endsymbol = '?'
    
    def computeWeightedGraph(self, G):
        """
        Public method that weights a graph G given its features
        @type G: a networkx graph
        @param G: the graph to weight

        @rtype: networkx graph
        @return: the weighted graph
        """
        dfeatures = self.getFeatures(G, nohash=True)#TODO mettere parametro no hash
        wg = nx.Graph(G)
        pairs = []
        for (k,v) in dfeatures.items():
            pairs.append((self.encodingToDag(k)[0],v))
            
        for (dag,frequency) in pairs:
            for u in dag.nodes():
                if wg.node[u].get('weight') is None:
                    wg.node[u]['weight']=frequency
                else:
                    wg.node[u]['weight']+=frequency
            
        return wg
            
        
    def encodingToDag(self, encode):
        """
        Public method that creates a DAG given its encoding
        @type encode: string
        @param encode: the DAG's encoding
        
        @rtype: triple (networkx.DiGraph,string,integer)
        @return: the triple that contains the created DAG, the left string to parse and the root's index
        """
        DAG=nx.DiGraph()
        start=encode.find(self.__startsymbol)
        if start==-1:
            start=sys.maxint
        end=encode.find(self.__endsymbol)
        if end==-1:
            end=sys.maxint
        conj=encode.find(self.__conjsymbol)
        if conj==-1:
            conj=sys.maxint
        minindex=min(start,end,conj)
        number=int(encode[:minindex])
        DAG.add_node(number)
        encode=encode[minindex:]
        if len(encode)!=0:
            if encode[0]==self.__startsymbol:
                encode=encode[1:]
                (childDAG,encodeleft,root)=self.encodingToDag(encode)
                childrenDAG=[childDAG]
                childrenRoot=[root]
                while(encodeleft[0]==self.__conjsymbol):
                    encodeleft=encodeleft[1:]
                    (childDAG,encodeleft,root)=self.encodingToDag(encodeleft)
                    childrenDAG.append(childDAG)
                    childrenRoot.append(root)
                if encodeleft[0]==self.__endsymbol:
                    encodeleft=encodeleft[1:]
                    compose=nx.DiGraph(DAG)
                    for g in childrenDAG:
                        compose=nx.compose(compose,g)
                    for r in childrenRoot:
                        compose.add_edge(number, r)
                    return (compose,encodeleft,number)
            else:
                return (DAG,encode,number)
        else:
            return (DAG,encode,number)
        
    def __normalization(self, feature_list):
        """
        Private method that normalize the feature vector if requested
        @type feature_list: Dictionary
        @param feature_list: Dictionary that represent the feature vector
        
        @rtype: Dictionary
        @return: The normalized feature vector
        """
        if self.normalization:
            total_norm = 0.0
        
            for value in feature_list.itervalues():
                total_norm += value*value
            
            normalized_feature_vector = {}
            sqrt_total_norm = math.sqrt( float(total_norm) )
            for (key,value) in feature_list.iteritems():
                normalized_feature_vector[key] = value/sqrt_total_norm
            return normalized_feature_vector
        else :
            return feature_list

    def computeGramTest(self,X,Y):
        """
        Public static method to compute the Gram matrix
        @type X: scipy.sparse.csr_matrix
        @param X: The instance-features matrix
        
        @rtype: numpy matrix
        @return: the Gram matrix
        """
        col1=X.indices
        col2=Y.indices
        col1,col2=self.__shrinkTwo(col1,col2)
        maximum=max(max(col1),max(col2))
        print maximum
        data=X.data
        row=[]
        v=0
        for i in range(1,X.indptr.shape[0]):
            for j in range(X.indptr[i]-X.indptr[i-1]):
                row.append(v)
            v+=1
        X=csr_matrix((data,(row,col1)),shape=(X.shape[0],maximum+1))
        data2=Y.data
        row2=[]
        v=0
        for i in range(1,Y.indptr.shape[0]):
            for j in range(Y.indptr[i]-Y.indptr[i-1]):
                row2.append(v)
            v+=1
        #print row2
        #print col2
        Y=csr_matrix((data2,(row2,col2)), shape=(Y.shape[0],maximum+1))
        print X.shape, Y.shape
        #col=self.__shrink(col) #needed to __shrink because if not can cause memory buffer overflow due to dot product implementation
        return X.dot(Y.T).todense()

    
    def computeGram(self,X):
        """
        Public static method to compute the Gram matrix
        @type X: list of scipy.sparse.csr_matrix
        @param X: The instance-features matrix indexed by height
        
        @rtype: numpy matrix
        @return: the Gram matrix
        """
        matrices=[]
        for mat in xrange(len(X)):
            col=X[mat].indices
            col=self.__shrink(col) #needed to __shrink because if not can cause memory buffer overflow due to dot product implementation
            data=X[mat].data
            row=[]
            v=0
            for i in range(1,X[mat].indptr.shape[0]):
                for j in range(X[mat].indptr[i]-X[mat].indptr[i-1]):
                    row.append(v)
                v+=1
            X[mat]=csr_matrix((data,(row,col)))
            matrices.append(X[mat].dot(X[mat].T).todense().tolist())
        return matrices
        
    def computeKernelMatrixTest(self,Graphs1,Graphs2):
        """
        Public static method to compute the Gram matrix
        @type X: scipy.sparse.csr_matrix
        @param X: The instance-features matrix
        
        @rtype: numpy matrix
        @return: the Gram matrix
        """
        X=self.transform(Graphs1)
        Y=self.transform(Graphs2)
        
        return self.computeGramTest(X,Y).tolist()
        
    def computeKernelMatrixTrain(self,Graphs):
        """
        Public static method to compute the Gram matrix
        @type X: scipy.sparse.csr_matrix
        @param X: The instance-features matrix
        
        @rtype: numpy matrix
        @return: the Gram matrix
        """
        X=self.transform(Graphs)
        return self.computeGram(X)

    @staticmethod
    def __shrink(col):
        """
        Private method that compresses the column index vector of some csr_matrix
        @type col: numpy array
        @param col: array that represents the column indexes of a csr_matrix
        
        @rtype: list
        @return: list of the compressed columns
        """
        minimums=np.sort(np.unique(col))
        for minindex in range(len(minimums)):
            currentmin=minimums[minindex]
            col[np.where(col==currentmin)]=minindex
        return col.tolist()
    @staticmethod
    def __shrinkTwo(col1,col2):
        """
        Private method that compresses the column index vector of some csr_matrix
        @type col: numpy array
        @param col: array that represents the column indexes of a csr_matrix
        
        @rtype: list
        @return: list of the compressed columns
        """
        minimums=np.sort(np.unique(np.concatenate([col1,col2])))
        for minindex in range(len(minimums)):
            currentmin=minimums[minindex]
            col1[np.where(col1==currentmin)]=minindex
            col2[np.where(col2==currentmin)]=minindex

        return col1.tolist(),col2.tolist()
    def __APHash(self, key):
        """
        Private method that computes the hash value for a given key
        @type key: string
        @param key: the string to digest
        
        @rtype: integer number
        @return: the digest
        """
        hashv = 0xAAAAAAAA
        for i in range(len(key)):
            if ((i & 1) == 0):
                hashv ^= ((hashv <<  7) ^ ord(key[i]) * (hashv >> 3))
            else:
                hashv ^= (~((hashv << 11) + ord(key[i]) ^ (hashv >> 5)))
        return hashv & self.__bitmask
    
    def __convert_to_sparse_matrix(self,feature_dict):
        """
        Private static method that convert the feature vector from dictionary to sparse matrix
        @type feature_dict: Dictionary
        @param feature_dict: a feature vector
        
        @rtype: scipy.sparse.csr_matrix
        @return: the feature vector in sparse form
        """
        if len(feature_dict) == 0:
            raise Exception('ERROR: something went wrong, empty feature_dict.')
        data = feature_dict.values()
        row, col = [], []
        for i, j in feature_dict.iterkeys():
            row.append( i )
            col.append( j )
        X = csr_matrix( (data,(row,col)), shape = (max(row)+1, max(col)+1))
        return X
    
    def __transform(self, instance_id , G_orig):
        """
        Private method that given a graph id and its representation computes the normalized feature vector
        @type instance_id: integer number
        @param instannce_id: a graph id
        
        @type G_orig: Networkx graph
        @param G_orig: a Networkx graph
        
        @rtype: Dictionary
        @return: The normalized feature vector
        """
        feature_list_depth=[]
        featuresAtDepth=self.getFeaturesDepth(G_orig)
        for i in xrange(len(featuresAtDepth)):
            feature_list_depth.append(defaultdict(lambda : defaultdict(float)))
            feature_list_depth[i].update({(instance_id,k):v for (k,v) in featuresAtDepth[i].items()})
            self.__normalization(feature_list_depth[i])
        return feature_list_depth
        
    def __transform_serial(self, G_list):
        """
        Private method that converts a networkx graph list into a instance-features matrix
        @type G_list: networkx graph generator
        @param G_list: list of the graph to convert
        
        @rtype: scipy.sparse.csr_matrix
        @return: the instance-features matrix
        """
        feature_dict_list=[{} for i in xrange(self.max_radius+1)]
        for instance_id , G in enumerate( G_list ):
            features=self.__transform( instance_id, G )
            for mat in xrange(len(features)):
                #print mat
                feature_dict_list[mat].update(features[mat])
        feature_dict_matrix=[self.__convert_to_sparse_matrix( feature_dict_list[i]) for i in xrange(len(feature_dict_list))]
        return feature_dict_matrix
    
    def transform(self, G_list, n_jobs = 1):
        """
        Public method that given a list of networkx graph it creates the sparse matrix (example, features) in parallel or serial
        @type G_list: networkx graph generator
        @param G_list: list of the graph to convert
        
        @type n_jobs: integer number
        @param n_jobs: number of parallel jobs
        
        @rtype: scipy.sparse.csr_matrix
        @return: the instance-features matrix
        """
        return self.__transform_serial(G_list)
    
    def getFeaturesDepth(self,G,nohash=False):
        """
        Public method that given a networkx graph G will create the dictionary representing its features according to the ST Kernel
        @type G: networkx graph
        @param G: the graph to extract features from
        
        @rtype: dictionary
        @return: the encoding-feature dictionary
        """
        Dict_features=[{} for i in xrange(self.max_radius+1)]
        for v in G.nodes():
            if G.node[v]['viewpoint']:
                if not G.graph['ordered']:
                    (DAG,maxLevel)=generateDAG(G, v, self.max_radius)
                    orderDAGvertices(DAG)
                else:
                    (DAG,maxLevel)=generateDAGOrdered(G, v, self.max_radius)

                MapNodeToProductionsID={} #k:list(unsigned)
                for u in DAG.nodes():
                    MapNodeToProductionsID[u]=[]
                MapNodetoFrequencies={} #k:list(int)
                for u in DAG.nodes():
                    MapNodetoFrequencies[u]=[]
                MapProductionIDtoSize={} #k:int
        
                for u in nx.topological_sort(DAG)[::-1]:
                    max_child_height=0
                    for child in DAG.successors(u):
                        child_height=0
                        if not MapNodeToProductionsID.get(child) is None:
                            child_height=len(MapNodeToProductionsID.get(child))
                        
                        if child_height > max_child_height:
                            max_child_height = child_height
                            
                    for depth in range(max_child_height+1):
                        hash_subgraph_code = 1
                        if depth==0:
                            enc=0
                            if nohash:
                                enc=str(u)
                            else:
                                enc=self.__APHash(DAG.node[u]['label'])
                            MapNodeToProductionsID[u].append(enc)
                            
                            frequency=0
                            if max_child_height==0:
                                frequency=maxLevel - DAG.node[u]['depth']
                            #Modified code----------------------
                            if Dict_features[depth].get(enc) is None:
                                Dict_features[depth][enc]=float(frequency+1.0)*math.sqrt(self.Lambda)
                            else:
                                Dict_features[depth][enc]+=float(frequency+1.0)*math.sqrt(self.Lambda)
                        
                            MapNodetoFrequencies[u].append(frequency)
                            MapProductionIDtoSize[enc]=1
                        else:
                            size=0
                            encoding=0
                            if nohash:
                                encoding=str(u)
                            else:
                                encoding=DAG.node[u]['label']
                            
                            vertex_label_id_list=[]#list[string]
                            min_freq_children=sys.maxint
                            
                            for child in DAG.successors(u):
                                size_map=len(MapNodeToProductionsID[child])
                                child_hash=MapNodeToProductionsID[child][min(size_map,depth)-1]
                                freq_child=MapNodetoFrequencies[child][min(size_map,depth)-1]
                                
                                if freq_child<min_freq_children:
                                    min_freq_children=freq_child
                                
                                vertex_label_id_list.append(str(child_hash))
                                size+=MapProductionIDtoSize[child_hash]
                            
                            vertex_label_id_list.sort()
                            encoding+=self.__startsymbol+vertex_label_id_list[0]
                            
                            for i in range(1,len(vertex_label_id_list)):
                                encoding+=self.__conjsymbol+vertex_label_id_list[i]
                            
                            encoding+=self.__endsymbol
                            if nohash:
                                hash_subgraph_code=encoding
                            else:
                                hash_subgraph_code=self.__APHash(encoding)
                            MapNodeToProductionsID[u].append(hash_subgraph_code)
                            size+=1
                            MapProductionIDtoSize[hash_subgraph_code]=size
                            
                            frequency = min_freq_children
                            MapNodetoFrequencies[u].append(frequency)
                            if Dict_features[depth].get(hash_subgraph_code) is None:
                                Dict_features[depth][hash_subgraph_code]=float(frequency+1.0)*math.sqrt(math.pow(self.Lambda,size))
                            else:
                                Dict_features[depth][hash_subgraph_code]+=float(frequency+1.0)*math.sqrt(math.pow(self.Lambda,size))
        return Dict_features
        
    def kernelFunction(self,Graph1, Graph2):
        #calculate the kernel between two graphs
        G_list = [Graph1, Graph2]
        vec=self.transform(G_list)
        return np.dot(vec[0], vec[1].T)
        #pass
