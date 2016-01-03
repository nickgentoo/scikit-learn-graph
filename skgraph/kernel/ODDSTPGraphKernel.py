from numpy.ma.testutils import approx
__author__ = "Riccardo Tesselli, Carlo Maria Massimo"
__date__ = "2/oct/2015"
__credits__ = ["Riccardo Tesselli", "Carlo Maria Massimo"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer = "Riccardo Tesselli, Carlo Maria Massimo"
__email__ = "riccardo.tesselli@gmail.com, cmassim@gmail.com"
__status__ = "Production"

from operator import itemgetter
from graphKernel import GraphKernel
from ..graph.GraphTools import drawGraph, generateDAG
from KernelTools import convert_to_sparse_matrix
import networkx as nx
import math
import sys
import numpy as np

class ODDSTPGraphKernel(GraphKernel):
    def __init__(self, r = 3, l = 1, normalization = True,version=1,show=False, ntype =0, nsplit =0):
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
        self.normalization_type = ntype
        self.split_normalization = nsplit
        self.show=show
        self.__startsymbol='!' #special symbols used in encoding
        self.__conjsymbol='#'
        self.__endsymbol='?'
        self.__contextsymbol='@'
        self.__oddkfeatsymbol='*'
        self.__version=version
    
    def transform(self, G_list, n_jobs = 1, approximated=True):
        """
        Public method that given a list of networkx graph it creates the sparse matrix (example, features) in parallel or serial
        @type G_list: networkx graph generator
        @param G_list: list of the graph to convert
        
        @type n_jobs: integer number
        @param n_jobs: number of parallel jobs
        
        @type approximated: boolean
        @param approximated: true if use a hash function with probable collisions during feature decomposition. False no collision guaranteed
        
        #TODO
        """
        if n_jobs is 1:
            return self.__transform_serial(G_list,approximated)
        else:
            return self.__transform_parallel(G_list, n_jobs,approximated)
     
    def __transform(self, G_orig, approximated=True):
        """
        Private method that given a graph id and its representation computes the normalized feature vector
        @type instance_id: integer number
        @param instannce_id: a graph id
        
        @type G_orig: Networkx graph
        @param G_orig: a Networkx graph
        
        #TODO
        """

        feature_list = {}
        
        if approximated:
            feature_list.update(self.getFeaturesApproximated(G_orig))
        else:
            feature_list.update(self.getFeaturesNoCollisions(G_orig))

        return self.__normalization(feature_list)
        
    def __transform_explicit(self, instance_id, G_orig, approximated=True):
        feature_list = {}

        if approximated:
            feature_list.update({(instance_id,k):v for (k,v) in self.getFeaturesApproximatedExplicit(G_orig).items()})
        else:
            feature_list.update({(instance_id,k):v for (k,v) in self.getFeaturesNoCollisionsExplicit(G_orig).items()})

        return feature_list
        
    def __transform_serial(self, G_list, approximated=True):
        """
        Private method that converts a networkx graph list into a instance-features matrix
        @type G_list: networkx graph generator
        @param G_list: list of the graph to convert
        
        @type approximated: boolean
        @param approximated: true if use a hash function with probable collisions during feature decomposition. False no collision guaranteed
        
        #TODO
        """
        list_dict=[]
        for G in G_list:
            if self.show:
                drawGraph(G)            
            list_dict.append(self.__transform(G, approximated))
        
        return list_dict
    
    def transform_serial_explicit(self,G_list,approximated=True):
        list_dict={}
        for instance_id, G in enumerate(G_list):
            if self.show:
                drawGraph(G)
            list_dict.update(self.__transform_explicit(instance_id,G, approximated))
        
        return convert_to_sparse_matrix(list_dict)
    
    def transform_serial_explicit_nomatrix(self, G_list, approximated=False):
        list_dict={}
        for instance_id, G in enumerate(G_list):
            if self.show:
                drawGraph(G)
            list_dict.update(self.__transform_explicit(instance_id, G, approximated))
        
        return list_dict
    
    def __transform_parallel(self,G_list, n_jobs,approximated=True,keepdictionary=False):
        """
        #TODO
        """
        print "TODO __transform_parallel"
        pass
    
    # TODO still with contexts
    def getFeaturesApproximated(self,G):#TODO usa xxhash lib con bitsize settabile
        """
        Public method that given a networkx graph G will create the dictionary representing its features according to the ST Kernel.
        The computation will use a hash function to encode a feature. There might be collisions
        @type G: networkx graph
        @param G: the graph to extract features from
        
        @type hashsize: integer
        @param hashsize: number of bits of the hash function to use
        
        #TODO
        """
        print "TODO implicit approximated ODDWCKernelPlus"
        pass
        Dict_features={}
        for v in G.nodes():
            (DAG,maxLevel)=generateDAG(G, v, self.max_radius)
            
            if self.show:
                drawGraph(DAG,v)
                
            MapNodeToProductionsID={} #k:list(int)
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
                        
                        increment=frequency+1
                        
                        if Dict_features.get(enc) is None:
                            Dict_features[enc]=([0,0,0,self.Lambda],{}) #4th element is lambda^size=lambda^1
                        if u==v:
                            Dict_features[enc][0][0]+=increment
                        Dict_features[enc][0][1]+=increment
                        
                        Dict_features[enc][0][2]+=DAG.node[u]['paths']*increment
                                                
                        MapNodetoFrequencies[u].append(frequency)
                        MapProductionIDtoSize[enc]=1
                    else:
                        size=0
                        encoding=DAG.node[u]['label']
                        
                        vertex_label_id_list=[]#list[int]
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
                        #size*=2 #TODO bug navarin
                        size+=1
                        MapProductionIDtoSize[encoding]=size
                        
                        frequency = min_freq_children
                        MapNodetoFrequencies[u].append(frequency)
                        
                        increment=frequency+1
                        
                        if Dict_features.get(encoding) is None:
                            Dict_features[encoding]=([0,0,0,math.pow(self.Lambda,size)],{})
                        if u==v:
                            Dict_features[encoding][0][0]+=increment
                        
                        Dict_features[encoding][0][1]+=increment
                        
                        Dict_features[encoding][0][2]+=DAG.node[u]['paths']*increment
                        
                        i=0
                        while i<len(vertex_label_id_list):
                            if not Dict_features[vertex_label_id_list[i]][1].get(encoding) is None:
                                while i+1<len(vertex_label_id_list) and vertex_label_id_list[i]==vertex_label_id_list[i+1]:
                                    i+=1 #skip. the children already has been set
                            else:
                                cont=1
                                while i+1<len(vertex_label_id_list) and vertex_label_id_list[i]==vertex_label_id_list[i+1]:
                                    cont+=1
                                    i+=1
                                Dict_features[vertex_label_id_list[i]][1][encoding]=cont #adding context to its children with the children frequency in its context
                            i+=1
        
        return Dict_features
    
    # TODO still with contexts
    def getFeaturesNoCollisions(self,G):
        """
        Public method that given a networkx graph G will create the dictionary representing its features according to the ST Kernel
        @type G: networkx graph
        @param G: the graph to extract features from
        
        @rtype: dictionary
        @return: the encoding-feature dictionary
        
        TODO
        """
        print "TODO implicit no collision ODDWCKernelPlus"
        pass
        Dict_features={}
        for v in G.nodes():
            (DAG,maxLevel)=generateDAG(G, v, self.max_radius)
            
            if self.show:
                drawGraph(DAG,v)
                
            MapNodeToProductionsID={} #k:list(string)
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
                        enc=DAG.node[u]['label']
                        MapNodeToProductionsID[u].append(enc)
                        
                        frequency=0
                        if max_child_height==0:
                            frequency=maxLevel - DAG.node[u]['depth']
                        
                        increment=frequency+1
                        
                        if Dict_features.get(enc) is None:
                            Dict_features[enc]=([0,0,0,self.Lambda],{}) #4th element is lambda^size=lambda^1
                        if u==v:
                            Dict_features[enc][0][0]+=increment
                        Dict_features[enc][0][1]+=increment
                        
                        Dict_features[enc][0][2]+=DAG.node[u]['paths']*increment
                                                
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
                        encoding+=self.__startsymbol+vertex_label_id_list[0]
                        
                        for i in range(1,len(vertex_label_id_list)):
                            encoding+=self.__conjsymbol+vertex_label_id_list[i]
                        
                        encoding+=self.__endsymbol
                        
                        MapNodeToProductionsID[u].append(encoding)
                        size+=1
                        MapProductionIDtoSize[encoding]=size
                        
                        frequency = min_freq_children
                        MapNodetoFrequencies[u].append(frequency)
                        
                        increment=frequency+1
                        
                        if Dict_features.get(encoding) is None:
                            Dict_features[encoding]=([0,0,0,math.pow(self.Lambda,size)],{})
                        if u==v:
                            Dict_features[encoding][0][0]+=increment
                        
                        Dict_features[encoding][0][1]+=increment
                        
                        Dict_features[encoding][0][2]+=DAG.node[u]['paths']*increment
                        
                        i=0
                        while i<len(vertex_label_id_list):
                            if not Dict_features[vertex_label_id_list[i]][1].get(encoding) is None:
                                while i+1<len(vertex_label_id_list) and vertex_label_id_list[i]==vertex_label_id_list[i+1]:
                                    i+=1 #skip. the children already has been set
                            else:
                                cont=1
                                while i+1<len(vertex_label_id_list) and vertex_label_id_list[i]==vertex_label_id_list[i+1]:
                                    cont+=1
                                    i+=1
                                Dict_features[vertex_label_id_list[i]][1][encoding]=cont #adding context to its children with the children frequency in its context
                            i+=1
                            
        return Dict_features
    
    def getFeaturesApproximatedExplicit(self,G):
        Dict_features={}
        if self.__version == 0:
            ODDK_Dict_features = {}

        for v in G.nodes():
            (DAG,maxLevel)=generateDAG(G, v, self.max_radius)
            
            if self.show:
                drawGraph(DAG,v)
                
            MapNodeToProductionsID={} #k:list(int)
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
                        MapProductionIDtoSize[enc]=1
                        
                        frequency=0
                        if max_child_height==0:
                            frequency=maxLevel - DAG.node[u]['depth']
                        
                        MapNodetoFrequencies[u].append(frequency)
                        
                        if self.__version==0:#add oddk feature
                            hashoddk=hash(self.__oddkfeatsymbol+str(enc))
                            if ODDK_Dict_features.get(hashoddk) is None:
                                ODDK_Dict_features[hashoddk]=0

                            weight = float(frequency+1.0)*math.sqrt(self.Lambda)
                            if self.normalization and self.normalization_type == 1:
                                weight = math.tanh(float(frequency+1.0))*math.tanh(math.sqrt(self.Lambda))

                            ODDK_Dict_features[hashoddk] += weight
                        
                        if u == v:
                            frequency = 0

                        # adding ST features to the dict (they are a subset of ST+)
                        if Dict_features.get(enc) is None:
                            Dict_features[enc] = 0

                        weight = float(frequency+1.0) * math.sqrt(self.Lambda)
                        if self.normalization and self.normalization_type == 1:
                            weight = math.tanh(float(frequency+1.0)) * math.tanh(math.sqrt(self.Lambda))
                        Dict_features[enc] +=  weight
                            
                    else:
                        size=0
                        #computing st feature
                        encoding=DAG.node[u]['label']
                        
                        vertex_label_id_list=[]#list[int]
                        min_freq_children=sys.maxint
                        
                        for child in DAG.successors(u):
                            size_map=len(MapNodeToProductionsID[child])
                            child_hash=MapNodeToProductionsID[child][min(size_map,depth)-1]
                            freq_child=MapNodetoFrequencies[child][min(size_map,depth)-1]
                            
                            if freq_child<min_freq_children:
                                min_freq_children=freq_child
                            
                            size_child=MapProductionIDtoSize[child_hash]
                            size+=size_child

                            vertex_label_id_list.append((child_hash,size_child))
                        
                        vertex_label_id_list.sort(key=itemgetter(0))
                        encoding+=self.__startsymbol+str(vertex_label_id_list[0][0])
                        
                        for i in range(1,len(vertex_label_id_list)):
                            encoding+=self.__conjsymbol+str(vertex_label_id_list[i][0])
                        
                        encoding+=self.__endsymbol
                        encoding=hash(encoding)
                        
                        MapNodeToProductionsID[u].append(encoding)
                        size+=1
                        MapProductionIDtoSize[encoding]=size
                        
                        frequency = min_freq_children
                        MapNodetoFrequencies[u].append(frequency)

                        if self.__version==0: #add oddk feature
                            oddkenc=hash(self.__oddkfeatsymbol+str(encoding))
                            if ODDK_Dict_features.get(oddkenc) is None:
                                ODDK_Dict_features[oddkenc]=0

                            weight = float(frequency+1.0)*math.sqrt(math.pow(self.Lambda,size))
                            if self.normalization and self.normalization_type == 1:
                                weight = math.tanh(float(frequency+1.0))*math.tanh(math.sqrt(math.pow(self.Lambda,size)))
                            ODDK_Dict_features[oddkenc] += weight

                        if u == v:
                            frequency = 0

                        # adding ST features to the dict (they are a subset of ST+)
                        if Dict_features.get(encoding) is None:
                            Dict_features[encoding] = 0

                        weight = float(frequency+1.0) * math.sqrt(math.pow(self.Lambda,size))
                        if self.normalization and self.normalization_type == 1:
                            weight = math.tanh(float(frequency+1.0)) * math.tanh(math.sqrt(math.pow(self.Lambda,size)))
                        Dict_features[encoding] +=  weight
                            
                        #extracting features st+
                        if len(vertex_label_id_list)>1: #if there's more than one child
                            successors=DAG.successors(u)
                            #extract ST+ features
                            for j in range(len(successors)):
                                for l in range(depth):
                                    branches=[]
                                    sizestplus=0
                                    for z in range(len(successors)):
                                        size_map=len(MapNodeToProductionsID[successors[z]])
                                        if j==z:
                                            child_hash=MapNodeToProductionsID[successors[z]][min(size_map,depth)-1]
                                            size_child=MapProductionIDtoSize[child_hash]
                                            sizestplus+=size_child
                                            branches.append((child_hash,size_child))
                                        else:
                                            if min(size_map,l)-1>=0:
                                                child_hash=MapNodeToProductionsID[successors[z]][min(size_map,l)-1]
                                                size_child=MapProductionIDtoSize[child_hash]
                                                sizestplus+=size_child
                                                branches.append((child_hash,size_child))
                                                
                                    branches.sort(key=itemgetter(0))
                                    
                                    encodingstplus=DAG.node[u]['label']
                                    encodingstplus +=self.__startsymbol+str(branches[0][0])
                         
                                    for i in range(1,len(branches)):
                                        encodingstplus += self.__conjsymbol+str(branches[i][0])
                                    
                                    encodingstplus+=self.__endsymbol
                                    encodingstplus=hash(encodingstplus)
                                    
                                    sizestplus+=1
                                    
                                    if self.__version==0: #add oddk st+ feature
                                        oddkenc=hash(self.__oddkfeatsymbol+str(encodingstplus))
                                        if ODDK_Dict_features.get(oddkenc) is None:
                                            ODDK_Dict_features[oddkenc]=0

                                        weight = float(frequency+1.0)*math.sqrt(math.pow(self.Lambda,sizestplus))
                                        if self.normalization and self.normalization_type == 1:
                                            weight = math.tanh(float(frequency+1.0))*math.tanh(math.sqrt(math.pow(self.Lambda,sizestplus)))
                                        ODDK_Dict_features[oddkenc] += weight
                                    
                                    if u==v:
                                        frequency = 0

                                    if Dict_features.get(encodingstplus) is None:
                                        Dict_features[encodingstplus] = 0

                                    weight= float(frequency+1.0) * math.sqrt(math.pow(self.Lambda, sizestplus))
                                    if self.normalization and self.normalization_type == 1:
                                        weight = math.tanh(float(frequency+1.0)) * math.tanh(math.sqrt(math.pow(self.Lambda, sizestplus)))
                                    Dict_features[encodingstplus] += weight
            
        if self.__version==0:

            # default case
            sdf = Dict_features
            osdf = ODDK_Dict_features

            # override default and apply split normalizatin if required
            if self.split_normalization:
                if self.normalization:
                    sdf = self.__normalization(Dict_features) 
                    osdf = self.__normalization(ODDK_Dict_features) 

            # merge the two feature dicts
            for (key,value) in osdf.iteritems():
                sdf[key] = value

            # again if normalization is required perform it (it won't affect the previous split normalization step)
            if self.normalization:
                return self.__normalization(sdf) 
            else:
                return sdf
        else:
            if self.normalization:
                return self.__normalization(Dict_features)
            else:
                return Dict_features
    
    def getFeaturesNoCollisionsExplicit(self,G):
        Dict_features={}
        for v in G.nodes():
            (DAG,maxLevel)=generateDAG(G, v, self.max_radius)
            
            if self.show:
                drawGraph(DAG,v)
                
            MapNodeToProductionsID={} #k:list(string)
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
                        enc=DAG.node[u]['label']
                            
                        MapNodeToProductionsID[u].append(enc)
                        MapProductionIDtoSize[enc]=1
                        
                        frequency=0
                        if max_child_height==0:
                            frequency=maxLevel - DAG.node[u]['depth']
                        
                        MapNodetoFrequencies[u].append(frequency)
                        
                        if u == v:
                            frequency = 0

                        # adding ST features to the dict (they are a subset of ST+)
                        if Dict_features.get(enc) is None:
                            Dict_features[enc] = 0
                        Dict_features[enc] += float(frequency+1.0) * math.sqrt(self.Lambda)
                        print "[depth==0] added feat: \"" + enc + "\" freq: " + str(frequency+1)

                    else:
                        size=0
                        encoding=DAG.node[u]['label']
                        
                        vertex_label_id_list=[]#list[string]
                        min_freq_children=sys.maxint
                        
                        #computing ST feature
                        for child in DAG.successors(u):
                            size_map=len(MapNodeToProductionsID[child])
                            child_hash=MapNodeToProductionsID[child][min(size_map,depth)-1]
                            freq_child=MapNodetoFrequencies[child][min(size_map,depth)-1]
                            
                            if freq_child<min_freq_children:
                                min_freq_children=freq_child
                            
                            size_child=MapProductionIDtoSize[child_hash]
                            size+=size_child

                            vertex_label_id_list.append((child_hash,size_child))
                        
                        vertex_label_id_list.sort(key=itemgetter(0))
                        encoding+=self.__startsymbol+vertex_label_id_list[0][0]
                        
                        for i in range(1,len(vertex_label_id_list)):
                            encoding+=self.__conjsymbol+vertex_label_id_list[i][0]
                        
                        encoding+=self.__endsymbol
                        
                        MapNodeToProductionsID[u].append(encoding)
                        size+=1
                        MapProductionIDtoSize[encoding]=size
                        
                        frequency = min_freq_children
                        MapNodetoFrequencies[u].append(frequency)
                        
                        if u == v:
                            frequency = 0

                        # adding ST features to the dict (they are a subset of ST+)
                        if Dict_features.get(encoding) is None:
                            Dict_features[encoding] = 0
                        Dict_features[encoding] += float(frequency+1.0) * math.sqrt(math.pow(self.Lambda,size))
                        print "[depth>0] added feat: \"" + encoding + "\""
                            
                        #extracting features st+
                        if len(vertex_label_id_list)>1: #if there's more than one child
                            successors=DAG.successors(u)
                            #extract ST+ features
                            for j in range(len(successors)):
                                for l in range(depth):
                                    branches=[]
                                    sizestplus=0
                                    for z in range(len(successors)):
                                        size_map=len(MapNodeToProductionsID[successors[z]])
                                        if j==z:
                                            child_hash=MapNodeToProductionsID[successors[z]][min(size_map,depth)-1]
                                            size_child=MapProductionIDtoSize[child_hash]
                                            sizestplus+=size_child
                                            branches.append((child_hash,size_child))
                                        else:
                                            if min(size_map,l)-1>=0:
                                                child_hash=MapNodeToProductionsID[successors[z]][min(size_map,l)-1]
                                                size_child=MapProductionIDtoSize[child_hash]
                                                sizestplus+=size_child
                                                branches.append((child_hash,size_child))
                                                
                                    branches.sort(key=itemgetter(0))
                                    
                                    encodingstplus=DAG.node[u]['label']
                                    encodingstplus +=self.__startsymbol+branches[0][0]
                        
                                    for i in range(1,len(branches)):
                                        encodingstplus += self.__conjsymbol+branches[i][0]
                                    
                                    encodingstplus+=self.__endsymbol
                                    
                                    sizestplus+=1

                                    if u==v:
                                        frequency = 0

                                    if Dict_features.get(encodingstplus) is None:
                                        Dict_features[encodingstplus] = 0
                                    Dict_features[encodingstplus] += float(frequency+1.0) * math.sqrt(math.pow(self.Lambda, sizestplus))
                                    print "added ST+ feat: \"" + encodingstplus + "\" l: " + str(l) + " j: " + str(j)
            print "---"
            
        for (k,v) in Dict_features.items():
            print k+": " + str(v)

        print len(Dict_features)

                                    
        return Dict_features
    
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

    def computeGramExplicit(self, g_it, approx=True, precomputed=None):
        if precomputed is None:
            precomputed=self.transform_serial_explicit(g_it, approximated=approx)
        return precomputed.dot(precomputed.T).todense().tolist()
    
    def computeGram(self, g_it, jobs=1, approx=True, precomputed=None):
        """
        Public static method to compute the Gram matrix
        @type g_it: networkx graph list
        @param g_it: graph instances
        
        @type jobs: integer
        @param jobs: number of parallel jobs. if -1 then number of jobs is maximum
        
        @type approx: boolean
        @param approx: if True then the approximated decomposition is used
        
        @rtype: numpy matrix
        @return: the Gram matrix
        """
        if precomputed is None:
            precomputed=self.transform(g_it, n_jobs=jobs, approximated=approx)
        gram=np.matrix(np.empty((len(precomputed),len(precomputed))))
        
        for i in xrange(len(precomputed)):
            for j in xrange(i,len(precomputed)):
                gram[i,j]=self.computeScore(precomputed[i], precomputed[j])
                gram[j,i]=gram[i,j]
                
        return gram.todense().tolist()
    
    def computeScore(self, Graph1, Graph2, poly=None):
        """
        #TODO
        """
        score=0
        
        if len(Graph1.keys())<len(Graph2.keys()):
            for f in Graph1.keys():
                if not Graph2.get(f) is None:
                    scorefeature=0
                    lambda_weight=Graph1[f][0][3]
                    scorefeature+=Graph1[f][0][0]*Graph2[f][0][0] #TODO commenta per bug. anche sotto
                    if self.__version==0:
                        scorefeature+=Graph1[f][0][1]*Graph2[f][0][1]
                    for c in Graph1[f][1].keys():
                        if not Graph2[f][1].get(c) is None:
                            scorefeature+=Graph1[c][0][2]*Graph2[c][0][2]*math.pow(Graph1[f][1][c],2)
                    scorefeature*=lambda_weight
                    if poly:
                        scorefeature=math.pow(scorefeature+1,poly)
                    score+=scorefeature
        else:
            for f in Graph2.keys():
                if not Graph1.get(f) is None:
                    scorefeature=0
                    lambda_weight=Graph1[f][0][3]
                    scorefeature+=Graph1[f][0][0]*Graph2[f][0][0]
                    if self.__version==0:
                        scorefeature+=Graph1[f][0][1]*Graph2[f][0][1]
                    for c in Graph2[f][1].keys():
                        if not Graph1[f][1].get(c) is None:
                            scorefeature+=Graph1[c][0][2]*Graph2[c][0][2]*math.pow(Graph1[f][1][c],2)
                    scorefeature*=lambda_weight
                    if poly:
                        scorefeature=math.pow(scorefeature+1,poly)
                    score+=scorefeature
                    
        return score

    def kernelFunction(self, Graph1, Graph2,approximated=True):
        """
        #TODO
        """
        listgraphdecompositions=self.transform([Graph1,Graph2],approximated=approximated)
        return self.computeScore(listgraphdecompositions[0], listgraphdecompositions[1])
    
    def computeKernelMatrixTrain(self, Graphs):
        return self.computeGramExplicit(Graphs)   

#if __name__=='__main__': #TODO converti in test
# g=nx.Graph()
# g.add_node(1,label='F')
# g.add_node(2,label='B')
# g.add_node(3,label='F')
# g.add_node(4,label='A')
# g.add_node(5,label='C')
# g.add_node(6,label='B')
# g.add_node(7,label='E')
# g.add_node(8,label='H')
# g.add_node(9,label='G')
# g.add_node(10,label='I')
# g.add_edge(1, 2)
# g.add_edge(1, 3)
# g.add_edge(1, 6)
# g.add_edge(1, 7)
# g.add_edge(2, 4)
# g.add_edge(3, 4)
# g.add_edge(4, 5)
# g.add_edge(7, 8)
# g.add_edge(8, 9)
# g.add_edge(9, 10)
# g.add_edge(6, 4)

#    g.add_node(1,label='A')
#    g.add_node(2,label='B')
#    g.add_node(3,label='C')
#    g.add_node(4,label='D')
#    #g.add_node(5,label='A')
#    g.add_edge(1, 2)
#    g.add_edge(2, 3)
#    g.add_edge(1, 4)
#    #g.add_edge(3, 4)
#    #g.add_edge(4, 5)
#    """
#    
#    
#    g.add_node(1,label='A')
#    g.add_node(2,label='B')
#    g.add_node(3,label='C')
#    g.add_node(4,label='D')
#    g.add_edge(1,2)
#    g.add_edge(1,3)
#    g.add_edge(2,4)
#    g.add_edge(3,4)
#    
#    
#    ke=ODDWCKernelPlus(r=2,normalization=False,l=1, version=1,show=0)
#    #print ke.computeGram([g,g], jobs=1, approx=True)
#    #print ke.computeGramExplicit([g,g], approx=False)
#    #print ke.computeGramExplicit([g,g], approx=True)
#    #print ke.getFeaturesNoCollisions(g) 
#    d= ke.getFeaturesNoCollisionsExplicit(g)
#    print d
#    print len(d)
#    print sum([v for k,v in d.items()])
#    l=[k for k in d.keys() if k[0]=='*']
#    l2=[v for k,v in d.items() if k[0]=='*']
#    print l
#    print len(l)
#    print sum(l2)
#    """
#    a=ke.getFeaturesNoCollisions(g)
#    b=ke.getFeaturesNoCollisionsExplicit(g)
#    
#    counter=0
#    for k in a.keys():
#        for c in a[k][1].keys():
#            if b.get(k+'@'+c) is None:
#                print (k,c)
#            else:
#                counter+=1
#                if not np.isclose(b.get(k+'@'+c),a[c][0][2]*a[k][1][c]*math.sqrt(a[k][0][3])):
#                    print ('non radicato',b.get(k+'@'+c),a[c][0][2]*a[k][1][c]*math.sqrt(math.pow(ke.Lambda,a[k][0][3])))
#                    print (k,c)
#        if a[k][0][0]!=0:
#            if b.get(k) is None:
#                print k
#            else:
#                counter+=1
#                if not np.isclose(b.get(k),a[k][0][0]*math.sqrt(a[k][0][3])):
#                    print ('radicato',b.get(k),a[k][0][0]*math.sqrt(math.pow(ke.Lambda,a[k][0][3])))
#                    print k
#    """
