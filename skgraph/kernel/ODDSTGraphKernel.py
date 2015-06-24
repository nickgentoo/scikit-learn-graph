# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:44:02 2015

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
from graphKernel import GraphKernel
from ..graph.GraphTools import generateDAG
from operator import itemgetter
from ..graph.GraphTools import drawGraph
from KernelTools import convert_to_sparse_matrix
import networkx as nx
import math
import sys
import numpy as np

class ODDSTGraphKernel(GraphKernel):
    """
    Class that implements the ODDKernel with ST kernel
    """
    
    class UniqueMap(object):
        """
        Inner class that creates a __map between elements and ascending unique values
        """
        def __init__(self):
            self.__counter=0
            self.__map={}
            
        def addElement(self,elem):
            if self.__map.get(elem) is None:
                self.__map[elem]=self.__counter
                self.__counter+=1
                
        def getElement(self,elem):
            return self.__map.get(elem)
    
    def __init__(self, r = 3, l = 1, normalization = True,show=False):
        """
        Constructor
        @type r: integer number
        @param r: ODDKernel Parameter
        
        @type l: number in (0,1]
        @param l: ODDKernel Parameter
        
        @type normalization: boolean
        @param normalization: True to normalize the feature vectors
        
        @type show: boolean
        @param show: If true shows graphs and DAGs during computation
        """
        self.Lambda=l
        self.max_radius=r
        self.normalization=normalization
        self.show=show
        self.__startsymbol='!' #special symbols used in encoding
        self.__conjsymbol='#'
        self.__endsymbol='?'
    
    def computeWeightedGraphFromFeatures(self,G,features):
        """
        #TODO
        """
        featureindexes=self.getFeaturesIndexes(G)
        wg=nx.Graph(G)
        for u in wg.nodes():
            wg.node[u]['weight']=0
        for f in featureindexes.keys():
            if not features.get(f) is None:
                value=features.get(f)
                DAG=self.encodingWithIndexesToDag(featureindexes[f])[0]
                for u in DAG.nodes():
                    wg.node[u]['weight']+=value
        return wg
                
    def computeWeightedGraph(self,G):
        """
        Public method that weights a graph G given its features in the encoding with nodes' indexes instead of labels
        @type G: a networkx graph
        @param G: the graph to weight
        
        @rtype: networkx graph
        @return: the weighted graph
        """
        dfeatures=self.getFeaturesNoCollisions(G, indexes=True)
        wg=nx.Graph(G)
        pairs=[]
        for (k,v) in dfeatures.items():
            pairs.append((self.encodingWithIndexesToDag(k)[0],v))
            
        for (dag,frequency) in pairs:
            for u in dag.nodes():
                if wg.node[u].get('weight') is None:
                    wg.node[u]['weight']=frequency
                else:
                    wg.node[u]['weight']+=frequency
        return wg

    def encodingWithLabelsToDag(self, encode, rootindex=0):
        """
        Public method that creates a DAG given its encoding with the nodes' labels
        @type encode: string
        @param encode: the DAG's encoding
        
        @rtype: triple (networkx.DiGraph,string,string)
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
        labelnode=encode[:minindex]
        DAG.add_node(rootindex, label=labelnode)
        encode=encode[minindex:]
        if len(encode)!=0:
            if encode[0]==self.__startsymbol:
                encode=encode[1:]
                (childDAG,encodeleft,root,indexnext)=self.encodingWithLabelsToDag(encode,rootindex+1)
                childrenDAG=[childDAG]
                childrenRoot=[root]
                while(encodeleft[0]==self.__conjsymbol):
                    encodeleft=encodeleft[1:]
                    (childDAG,encodeleft,root,indexnext)=self.encodingWithLabelsToDag(encodeleft,indexnext)
                    childrenDAG.append(childDAG)
                    childrenRoot.append(root)
                if encodeleft[0]==self.__endsymbol:
                    encodeleft=encodeleft[1:]
                    compose=nx.DiGraph(DAG)
                    for g in childrenDAG:
                        compose=nx.compose(compose,g)
                    for r in childrenRoot:
                        compose.add_edge(rootindex, r)
                    return (compose,encodeleft,rootindex,indexnext)
            else:
                return (DAG,encode,rootindex,rootindex+1)
        else:
            return (DAG,encode,rootindex,rootindex+1)
                    
    def encodingWithIndexesToDag(self, encode):
        """
        Public method that creates a DAG given its encoding with the nodes' index
        @type encode: string
        @param encode: the DAG's encoding
        
        @rtype: triple (networkx.DiGraph,string,string)
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
                (childDAG,encodeleft,root)=self.encodingWithIndexesToDag(encode)
                childrenDAG=[childDAG]
                childrenRoot=[root]
                while(encodeleft[0]==self.__conjsymbol):
                    encodeleft=encodeleft[1:]
                    (childDAG,encodeleft,root)=self.encodingWithIndexesToDag(encodeleft)
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
            return dict(feature_list)
            
    def computeKernelMatrixTrain(self,Graphs):
        return self.computeGram(Graphs)

    def computeGram(self, g_it, jobs=-1, approx=True, precomputed=None):
        """
        Public static method to compute the Gram matrix
        @type g_it: networkx graph list
        @param g_it: graph instances
        
        @type jobs: integer
        @param jobs: number of parallel jobs. if -1 then number of jobs is maximum
        
        @type approx: boolean
        @param approx: if True then the approximated decomposition is used
        
        @type precomputed: csr_sparse matrix
        @param precomputed: precomputed instance-features matrix
        
        @rtype: numpy matrix
        @return: the Gram matrix
        """
        if precomputed is None:
            precomputed=self.transform(g_it, n_jobs=jobs, approximated=approx)
        return precomputed.dot(precomputed.T).todense().tolist()

    def computeGramTest(self,X,Y,jobs=-1, approx=True, precomputed=None):
        """
        TODO TEST
        Public static method to compute the Gram matrix
        @type X: scipy.sparse.csr_matrix
        @param X: The instance-features matrix
        
        @rtype: numpy matrix
        @return: the Gram matrix
        """
        if precomputed is None:
            precomputed1=self.transform(X, n_jobs=jobs, approximated=approx)
            precomputed2=self.transform(X, n_jobs=jobs, approximated=approx)

        return precomputed1.dot(precomputed2.T).todense().tolist()
    def __transform(self, instance_id , G_orig, approximated=True, MapEncToId=None):
        """
        Private method that given a graph id and its representation computes the normalized feature vector
        @type instance_id: integer number
        @param instannce_id: a graph id
        
        @type G_orig: Networkx graph
        @param G_orig: a Networkx graph
        
        @type MapEncToId: self.UniqueMap
        @param MapEncToId: Map between feature's encodings and integer values
        
        @rtype: Dictionary
        @return: The normalized feature vector
        """
        #feature_list = defaultdict(lambda : defaultdict(float))
        feature_list={}
        if approximated:
            feature_list.update({(instance_id,k):v for (k,v) in self.getFeaturesApproximated(G_orig,MapEncToId).items()})
        else:
            feature_list.update({(instance_id,k):v for (k,v) in self.getFeaturesNoCollisions(G_orig,MapEncToId).items()})
        return self.__normalization(feature_list)
        
    def __transform_serial(self, G_list, approximated=True,keepdictionary=False):
        """
        Private method that converts a networkx graph list into a instance-features matrix
        @type G_list: networkx graph generator
        @param G_list: list of the graph to convert
        
        @type approximated: boolean
        @param approximated: true if use a hash function with probable collisions during feature decomposition. False no collision guaranteed
        
        @type keepdictionary: boolean
        @param keepdictionary: True if the instance-feature matrix is kept as a dictionary. Else is a csr_matrix
        
        @rtype: scipy.sparse.csr_matrix
        @return: the instance-features matrix
        """
        feature_dict={}
        MapEncToId=None
        if not keepdictionary:
            MapEncToId=self.UniqueMap()
        for instance_id , G in enumerate( G_list ):
            if self.show:
                drawGraph(G)
            
            feature_dict.update(self.__transform( instance_id, G, approximated, MapEncToId))
        if keepdictionary:
            return (convert_to_sparse_matrix( feature_dict, MapEncToId ),feature_dict)
        else:
            return convert_to_sparse_matrix( feature_dict, MapEncToId )
    
    
    def transform(self, G_list, n_jobs = -1, approximated=True, keepdictionary=False):
        """
        Public method that given a list of networkx graph it creates the sparse matrix (example, features) in parallel or serial
        @type G_list: networkx graph generator
        @param G_list: list of the graph to convert
        
        @type n_jobs: integer number
        @param n_jobs: number of parallel jobs
        
        @type approximated: boolean
        @param approximated: true if use a hash function with probable collisions during feature decomposition. False no collision guaranteed
        
        @type keepdictionary: boolean
        @param keepdictionary: True if the instance-feature matrix is kept as a dictionary. Else is a csr_matrix
        
        @rtype: scipy.sparse.csr_matrix
        @return: the instance-features matrix
        """
        if n_jobs is 1:
            return self.__transform_serial(G_list,approximated,keepdictionary)
        else:
            print "WARNING: parallel calculation not implemented"
            return self.__transform_serial(G_list,approximated,keepdictionary)

    def getFeaturesIndexes(self,G):
        """
        Public method that given a networkx graph G will create the dictionary representing its encoded features and the corresponding index nodes
        @type G: networkx graph
        @param G: the graph to extract features from
        
        @rtype: dictionary
        @return: the encoding-feature dictionary
        """
        Dict_features={}
        for v in G.nodes():
            DAG=generateDAG(G, v, self.max_radius)[0]
            
            if self.show:
                drawGraph(DAG,v)
                
            MapNodeToProductionsID={}
            MapNodeToProductionsIDInd={}
            for u in DAG.nodes():
                MapNodeToProductionsID[u]=[]
                MapNodeToProductionsIDInd[u]=[]
    
            for u in nx.topological_sort(DAG)[::-1]:
                max_child_height=0
                for child in DAG.successors(u):
                    child_height=len(MapNodeToProductionsID.get(child))
                    
                    if child_height > max_child_height:
                        max_child_height = child_height
                        
                for depth in range(max_child_height+1):
                    if depth==0:
                        enc=DAG.node[u]['label']
                        encind=str(u)
                        
                        MapNodeToProductionsID[u].append(enc)
                        MapNodeToProductionsIDInd[u].append(encind)
                        
                        if Dict_features.get(enc) is None:
                            Dict_features[enc]=encind
                        
                    else:
                        encoding=DAG.node[u]['label']
                        encodingind=str(u)
                        
                        vertex_label_id_list=[]
                        
                        for child in DAG.successors(u):
                            size_map=len(MapNodeToProductionsID[child])
                            child_hash=MapNodeToProductionsID[child][min(size_map,depth)-1]
                            child_hashind=MapNodeToProductionsIDInd[child][min(size_map,depth)-1]
                            
                            vertex_label_id_list.append((child_hash,child_hashind))

                        
                        vertex_label_id_list.sort(key=itemgetter(0))
                        encoding+=self.__startsymbol+vertex_label_id_list[0][0]
                        encodingind+=self.__startsymbol+vertex_label_id_list[0][1]
                        
                        for i in range(1,len(vertex_label_id_list)):
                            encoding+=self.__conjsymbol+vertex_label_id_list[i][0]
                            encodingind+=self.__conjsymbol+vertex_label_id_list[i][1]
                        
                        encoding+=self.__endsymbol
                        encodingind+=self.__endsymbol
                        
                        MapNodeToProductionsID[u].append(encoding)
                        MapNodeToProductionsIDInd[u].append(encodingind)
                        
                        if Dict_features.get(encoding) is None:
                            Dict_features[encoding]=encodingind
                        
        return Dict_features
    
    def getFeaturesNoCollisions(self,G,MapEncToId=None,indexes=False):
        """
        Public method that given a networkx graph G will create the dictionary representing its features according to the ST Kernel
        @type G: networkx graph
        @param G: the graph to extract features from
        
        @type MapEncToId: self.UniqueMap
        @param MapEncToId: Map between feature's encodings and integer values
        
        @type indexes: boolean
        @param indexes: if True the feature is encoded using the nodes' index rather than theirs labels
        
        @rtype: dictionary
        @return: the encoding-feature dictionary
        """
        Dict_features={}
        for v in G.nodes():
            (DAG,maxLevel)=generateDAG(G, v, self.max_radius)
            
            if self.show:
                drawGraph(DAG,v)
                
            MapNodeToProductionsID={} #k:list(unsigned)
            MapNodetoFrequencies={} #k:list(int)
            for u in DAG.nodes():
                MapNodeToProductionsID[u]=[]
                MapNodetoFrequencies[u]=[]
            MapProductionIDtoSize={} #k:int
    
            for u in nx.topological_sort(DAG)[::-1]:
                max_child_height=0
                for child in DAG.successors(u):
                    child_height=0
                    child_height=len(MapNodeToProductionsID.get(child))
                    
                    if child_height > max_child_height:
                        max_child_height = child_height
                 
                for depth in range(max_child_height+1):
                    if depth==0:
                        if not indexes:
                            enc=DAG.node[u]['label']
                        else:
                            enc=str(u)
                        
                        MapNodeToProductionsID[u].append(enc)
                        
                        frequency=0
                        if max_child_height==0:
                            frequency=maxLevel - DAG.node[u]['depth']
                        
                        if Dict_features.get(enc) is None:
                            Dict_features[enc]=float(frequency+1.0)*math.sqrt(self.Lambda)
                        else:
                            Dict_features[enc]+=float(frequency+1.0)*math.sqrt(self.Lambda)
                        
                        if not MapEncToId is None:
                            MapEncToId.addElement(enc)
                        
                        MapNodetoFrequencies[u].append(frequency)
                        MapProductionIDtoSize[enc]=1
                    else:
                        size=0
                        if not indexes:
                            encoding=DAG.node[u]['label']
                        else:
                            encoding=str(u)
                        
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
                        encoding+=self.__startsymbol+vertex_label_id_list[0]
                        
                        for i in range(1,len(vertex_label_id_list)):
                            encoding+=self.__conjsymbol+vertex_label_id_list[i]
                        
                        encoding+=self.__endsymbol
                        
                        MapNodeToProductionsID[u].append(encoding)
                        size+=1
                        MapProductionIDtoSize[encoding]=size
                        
                        frequency = min_freq_children
                        MapNodetoFrequencies[u].append(frequency)
                        
                        if Dict_features.get(encoding) is None:
                            Dict_features[encoding]=float(frequency+1.0)*math.sqrt(math.pow(self.Lambda,size))
                        else:
                            Dict_features[encoding]+=float(frequency+1.0)*math.sqrt(math.pow(self.Lambda,size))
                        
                        if not MapEncToId is None:
                            MapEncToId.addElement(encoding)    
                        
        return Dict_features
    
    def getFeaturesApproximated(self,G,MapEncToId=None):#TODO usa xxhash lib con bitsize settabile
        """
        Public method that given a networkx graph G will create the dictionary representing its features according to the ST Kernel.
        The computation will use a hash function to encode a feature. There might be collisions
        @type G: networkx graph
        @param G: the graph to extract features from
        
        @type hashsize: integer
        @param hashsize: number of bits of the hash function to use
        
        @type MapEncToId: self.UniqueMap
        @param MapEncToId: Map between feature's encodings and integer values
        
        @rtype: dictionary
        @return: the encoding-feature dictionary
        """
        Dict_features={}
        for v in G.nodes():
            (DAG,maxLevel)=generateDAG(G, v, self.max_radius)
            
            if self.show:
                drawGraph(DAG,v)
                
            MapNodeToProductionsID={} #k:list(unsigned)
            MapNodetoFrequencies={} #k:list(int)
            for u in DAG.nodes():
                MapNodeToProductionsID[u]=[]
                MapNodetoFrequencies[u]=[]
            MapProductionIDtoSize={} #k:int
    
            for u in nx.topological_sort(DAG)[::-1]:
                max_child_height=0
                for child in DAG.successors(u):
                    child_height=len(MapNodeToProductionsID.get(child))
                    
                    if child_height > max_child_height:
                        max_child_height = child_height
                        
                for depth in range(max_child_height+1):
                    if depth==0:
                        enc=hash(DAG.node[u]['label'])
                        
                        MapNodeToProductionsID[u].append(enc)
                        
                        frequency=0
                        if max_child_height==0:
                            frequency=maxLevel - DAG.node[u]['depth']
                        
                        if Dict_features.get(enc) is None:
                            Dict_features[enc]=float(frequency+1.0)*math.sqrt(self.Lambda)
                        else:
                            Dict_features[enc]+=float(frequency+1.0)*math.sqrt(self.Lambda)
                        
                        if not MapEncToId is None:
                            MapEncToId.addElement(enc)
                        
                        MapNodetoFrequencies[u].append(frequency)
                        MapProductionIDtoSize[enc]=1
                    else:
                        size=0
                        encoding=DAG.node[u]['label']
                        
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
                        #size*=2 #TODO bug Navarin
                        size+=1
                        MapProductionIDtoSize[encoding]=size
                        
                        frequency = min_freq_children
                        MapNodetoFrequencies[u].append(frequency)
                        
                        if Dict_features.get(encoding) is None:
                            Dict_features[encoding]=float(frequency+1.0)*math.sqrt(math.pow(self.Lambda,size))
                        else:
                            Dict_features[encoding]+=float(frequency+1.0)*math.sqrt(math.pow(self.Lambda,size))
                        
                        if not MapEncToId is None:
                            MapEncToId.addElement(encoding)    

        return Dict_features
        
    def kernelFunction(self,Graph1, Graph2):
        """
        #TODO
        """
        G_list = [Graph1, Graph2]
        X=self.transform(G_list, n_jobs=1)
        row1=X[0]
        row2=X[1]
        return row1.dot(row2.T)[0,0]