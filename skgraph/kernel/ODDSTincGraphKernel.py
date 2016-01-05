__author__ = "Carlo Maria Massimo"
__date__ = "17/dec/2015"
__credits__ = ["Carlo Maria Massimo"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer = "Carlo Maria Massimo"
__status__ = "Production"

from operator import itemgetter
from graphKernel import GraphKernel
from ..graph.GraphTools import drawGraph, generateDAG
from KernelTools import convert_to_sparse_matrix
import networkx as nx
import math
import sys
import numpy as np
from cvxopt import matrix

class ODDSTincGraphKernel(GraphKernel):
    def __init__(self, r =3, l =1, normalization =True, version =1, ntype =0, nsplit =0, kernels =[]):
        """
        Constructor
        @type r: integer number
        @param r: ODDKernel Parameter
        
        @type l: number in (0,1]
        @param l: ODDKernel Parameter
        
        @type normalization: boolean
        @param normalization: True to normalize the feature vectors
        
        """
        
        if kernels == []:
            raise Exception("'kernels' parameter cannot be empty. At least one entry among [ST|STC|STP|STPC] should be provided.")
        else:
            self.kernels = kernels

        self.Lambda=l
        self.max_radius=r
        self.normalization=normalization
        self.normalization_type = ntype
        self.split_normalization = nsplit
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
     
    def __transform_explicit(self, instance_id, G_orig, approximated =True):
        feature_lists = dict.fromkeys(self.kernels, {})

#        if approximated:
        feature_maps = self.getFeaturesApproximatedExplicit(G_orig)
        for key in self.kernels:
            for i, phi in feature_maps[key].items():
                feature_lists[key][i] = {(instance_id,k):v for (k,v) in phi.items()}
#        else:
#            feature_list.update({(instance_id,k):v for (k,v) in self.getFeaturesNoCollisionsExplicit(G_orig).items()})

        return feature_lists
        
    def transform_serial_explicit(self, G_list, approximated =True):
        feature_matrices = []
        feature_lists = dict.fromkeys(self.kernels, {})

        for instance_id, G in enumerate(G_list):
            tmpfeats = self.__transform_explicit(instance_id, G, approximated)
            for key in self.kernels:
                for i, phi in tmpfeats[key].items():
                    if i in feature_lists[key].keys():
                        feature_lists[key][i].update(phi)
                    else:
                        feature_lists[key][i] = phi

        for key in self.kernels:
            for i, phi in feature_lists[key].items():
                feature_matrices.append(convert_to_sparse_matrix(phi))

        return feature_matrices
    
    # DEPRECATED
    def transform_serial_explicit_nomatrix(self, G_list, approximated=False):
        list_dict={}
        for instance_id, G in enumerate(G_list):
            list_dict.update(self.__transform_explicit(instance_id, G, approximated))
        
        return list_dict
    
    def getFeaturesApproximatedExplicit(self, G):
        # Inits feature maps (the \phi{}s for the selected kernels)

        feature_maps = dict.fromkeys(self.kernels, {i:{} for i in range(self.max_radius+1)})

        if self.__version==0:
            ODDK_Dict_features = {i:{} for i in range(self.max_radius+1)}

        for v in G.nodes():
            (DAG,maxLevel)=generateDAG(G, v, self.max_radius)
            
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
                            if ODDK_Dict_features[depth].get(hashoddk) is None:
                                ODDK_Dict_features[depth][hashoddk]=0

                            # default weighting:
                            weight = float(frequency+1.0)*math.sqrt(self.Lambda)

                            # tanh normalization
                            if self.normalization and self.normalization_type == 1:
                                weight = math.tanh(float(frequency+1.0))*math.tanh(math.sqrt(self.Lambda))

                            ODDK_Dict_features[depth][hashoddk] += weight
                        
                        
                        weight = math.sqrt(self.Lambda)
                        if self.normalization and self.normalization_type == 1:
                            weight = math.tanh(math.sqrt(self.Lambda))


                        if 'ODDST' in self.kernels:
                            if feature_maps['ODDST'][depth].get(enc) is None:
                                feature_maps['ODDST'][depth][enc] = 0
                            feature_maps['ODDST'][depth][enc] += weight
                        if 'ODDSTP' in self.kernels:
                            if feature_maps['ODDSTP'][depth].get(enc) is None:
                                feature_maps['ODDSTP'][depth][enc] = 0
                            feature_maps['ODDSTP'][depth][enc] += weight

                        if u==v:
                            if 'ODDSTPC' in self.kernels:
                                if feature_maps['ODDSTPC'][depth].get(enc) is None:
                                    feature_maps['ODDSTPC'][depth][enc] = 0
                                feature_maps['ODDSTPC'][depth][enc] += weight
                            if 'ODDSTC' in self.kernels:
                                if feature_maps['ODDSTC'][depth].get(enc) is None:
                                    feature_maps['ODDSTC'][depth][enc] = 0
                                feature_maps['ODDSTC'][depth][enc] += weight

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
                            if ODDK_Dict_features[depth].get(oddkenc) is None:
                                ODDK_Dict_features[depth][oddkenc]=0

                            weight = float(frequency+1.0)*math.sqrt(math.pow(self.Lambda,size))
                            if self.normalization and self.normalization_type == 1:
                                weight = math.tanh(float(frequency+1.0))*math.tanh(math.sqrt(math.pow(self.Lambda,size)))

                            ODDK_Dict_features[depth][oddkenc] += weight
                        
                        # Adds context ST features
                        i=0
                        while i<len(vertex_label_id_list):
                            size_child = vertex_label_id_list[i][1]
                            weight = math.sqrt(math.pow(self.Lambda, size_child))
                            encodingfin = str(vertex_label_id_list[i][0])+self.__contextsymbol+str(encoding)
                            encodingfin = hash(encodingfin)

                            full_weight = weight*DAG.node[u]['paths']*(frequency+1)
                            if self.normalization and self.normalization_type == 1:
                                full_weight = math.tanh(weight)*math.tanh(DAG.node[u]['paths']*(frequency+1))

                            if 'ODDSTC' in self.kernels:
                                if feature_maps['ODDSTC'][depth].get(encodingfin) is None:
                                    feature_maps['ODDSTC'][depth][encodingfin] = 0
                                feature_maps['ODDSTC'][depth][encodingfin] += full_weight
                            if 'ODDSTPC' in self.kernels:
                                if feature_maps['ODDSTPC'][depth].get(encodingfin) is None:
                                    feature_maps['ODDSTPC'][depth][encodingfin] = 0
                                feature_maps['ODDSTPC'][depth][encodingfin] += full_weight
                            i+=1

                        weight = math.sqrt(math.pow(self.Lambda,size))
                        if self.normalization and self.normalization_type == 1:
                            weight = math.tanh(math.sqrt(math.pow(self.Lambda,size)))

                        if 'ODDST' in self.kernels:
                            if feature_maps['ODDST'][depth].get(encoding) is None:
                                feature_maps['ODDST'][depth][encoding] = 0
                            feature_maps['ODDST'][depth][encoding] += weight
                        if 'ODDSTP' in self.kernels:
                            if feature_maps['ODDSTP'][depth].get(encoding) is None:
                                feature_maps['ODDSTP'][depth][encoding] = 0
                            feature_maps['ODDSTP'][depth][encoding] += weight

                        if u==v:
                            if 'ODDSTC' in self.kernels:
                                if feature_maps['ODDSTC'][depth].get(encoding) is None:
                                    feature_maps['ODDSTC'][depth][encoding] = 0
                                feature_maps['ODDSTC'][depth][encoding] += weight
                            if 'ODDSTPC' in self.kernels:
                                if feature_maps['ODDSTPC'][depth].get(encoding) is None:
                                    feature_maps['ODDSTPC'][depth][encoding] = 0
                                feature_maps['ODDSTPC'][depth][encoding] += weight
                           
                        # Extracting ST+ features
                        if len(vertex_label_id_list)>1: #if there's more than one child
                            successors=DAG.successors(u)
                            # Extracts ST+ features
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
                                        if ODDK_Dict_features[depth].get(oddkenc) is None:
                                            ODDK_Dict_features[depth][oddkenc]=0

                                        weight = float(frequency+1.0)*math.sqrt(math.pow(self.Lambda,sizestplus))
                                        if self.normalization and self.normalization_type == 1:
                                            weight = math.tanh(float(frequency+1.0))*math.tanh(math.sqrt(math.pow(self.Lambda,sizestplus)))

                                        ODDK_Dict_features[depth][oddkenc] += weight
                                    
                                    # Adds context ST+ features
                                    i=0
                                    while i<len(branches):
                                        size_child=branches[i][1]
                                        weight=math.sqrt(math.pow(self.Lambda,size_child))
                                        encodingfin=str(branches[i][0])+self.__contextsymbol+str(encodingstplus)
                                        encodingfin=hash(encodingfin)

                                        full_weight = weight*DAG.node[u]['paths']*(frequency+1)
                                        if self.normalization and self.normalization_type == 1:
                                            full_weight = math.tanh(weight)*math.tanh(DAG.node[u]['paths']*(frequency+1))

                                        if 'ODDSTPC' in self.kernels:
                                            if feature_maps['ODDSTPC'][depth].get(encodingfin) is None:
                                                feature_maps['ODDSTPC'][depth][encodingfin]=0
                                            feature_maps['ODDSTPC'][depth][encodingfin] += full_weight

                                        i+=1

                                    if u==v:
                                        weight = math.sqrt(math.pow(self.Lambda,sizestplus))
                                        if self.normalization and self.normalization_type == 1:
                                            weight = math.tanh(math.sqrt(math.pow(self.Lambda,sizestplus)))

                                        if 'ODDSTP' in self.kernels:
                                            if feature_maps['ODDSTP'][depth].get(encodingstplus) is None:
                                                feature_maps['ODDSTP'][depth][encodingstplus]=0
                                            feature_maps['ODDSTP'][depth][encodingstplus] += weight
 
        processed_feat_maps = dict.fromkeys(self.kernels, {})
        if self.__version==0:

            for key in self.kernels:
                for i, phi in feature_maps[key].items():
                    if not len(phi)==0:

                        # ODDST has no version 0 kernel
                        if key != 'ODDST':
                            # default case
                            sdf = phi
                            osdf = ODDK_Dict_features[i]

                            # override default and apply split normalizatin if required
                            if self.split_normalization:
                                if self.normalization:
                                    sdf = self.__normalization(phi) 
                                    osdf = self.__normalization(ODDK_Dict_features[i]) 

                            # merge the two feature dicts
                            for (key,value) in osdf.iteritems():
                                sdf[key] = value

                            # again if normalization is required perform it (it won't affect the previous split normalization step)
                            if self.normalization:
                                processed_feat_maps[key][i] = self.__normalization(sdf) 
                            else:
                                processed_feat_maps[key][i] = sdf
        else:
            for key in self.kernels:
                for i, phi in feature_maps[key].items():
                    if self.normalization:
                        processed_feat_maps[key][i] = self.__normalization(phi)
                    else:
                        processed_feat_maps[key][i] = phi

        return processed_feat_maps
    
    # TODO implement
    def getFeaturesNoCollisionsExplicit(self,G):
        Dict_features={}
        for v in G.nodes():
            (DAG,maxLevel)=generateDAG(G, v, self.max_radius)
            
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
                        
                        if self.__version==0:#add oddk feature
                            oddkenc=self.__oddkfeatsymbol+enc
                            if Dict_features.get(oddkenc) is None:
                                Dict_features[oddkenc]=0
                            Dict_features[oddkenc]+=float(frequency+1.0)*math.sqrt(self.Lambda)
                        
                        if u==v:
                            if Dict_features.get(enc) is None:
                                Dict_features[enc]=0
                            Dict_features[enc]+=math.sqrt(self.Lambda)
#                            print "[ST depth==0 && u==v] added feat: \"" + enc + "\""

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
                        
                        if self.__version==0: #add oddk feature
                            oddkenc=self.__oddkfeatsymbol+encoding
                            if Dict_features.get(oddkenc) is None:
                                Dict_features[oddkenc]=0
                            Dict_features[oddkenc]+=float(frequency+1.0)*math.sqrt(math.pow(self.Lambda,size))
                        
                        #adding context features
                        i=0
                        while i<len(vertex_label_id_list):
                            size_child=vertex_label_id_list[i][1]
                            weight=math.sqrt(math.pow(self.Lambda,size_child))
                            encodingfin=vertex_label_id_list[i][0]+self.__contextsymbol+encoding
                            if Dict_features.get(encodingfin) is None:
                                Dict_features[encodingfin]=0
                            Dict_features[encodingfin]+=weight*DAG.node[u]['paths']*(frequency+1)
                            i+=1
#                            print "[ST context] added feat: \"" + encodingfin + "\""
                        if u==v:
                            if Dict_features.get(encoding) is None:
                                Dict_features[encoding]=0
                            Dict_features[encoding]+=math.sqrt(math.pow(self.Lambda,size))
#                            print "[ST depth>0 & u==v] added feat: \"" + encoding + "\""
                            
                        #extracting features st+
                        if len(vertex_label_id_list)>1: #if there's more than one child
                            successors=DAG.successors(u)
                            #extract ST+ features
                            for j in range(len(successors)):
                                for l in range(depth):
#                                    print "*** node (j): " + str(j) + " depth (l): " + str(l) + " ***" 
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
                                    
                                    if self.__version==0: #add oddk st+ feature
                                        oddkenc=self.__oddkfeatsymbol+encodingstplus
                                        if Dict_features.get(oddkenc) is None:
                                            Dict_features[oddkenc]=0
                                        Dict_features[oddkenc]+=float(frequency+1.0)*math.sqrt(math.pow(self.Lambda,sizestplus))
                                    
                                    #add context st+ features
                                    i=0
                                    while i<len(branches):
                                        size_child=branches[i][1]
                                        weight=math.sqrt(math.pow(self.Lambda,size_child))
                                        encodingfin=branches[i][0]+self.__contextsymbol+encodingstplus
                                        if Dict_features.get(encodingfin) is None:
                                            Dict_features[encodingfin]=0
                                        Dict_features[encodingfin]+=weight*DAG.node[u]['paths']*(frequency+1)
                                        i+=1
#                                        print "[ST+ context] added feat: \"" + encodingfin + "\""
                                    if u==v:
                                        if Dict_features.get(encodingstplus) is None:
                                            Dict_features[encodingstplus]=0
                                        Dict_features[encodingstplus]+=math.sqrt(math.pow(self.Lambda,sizestplus))
#                                        print "[ST+ u==v] added feat: \"" + encodingstplus + "\""

#                                    print "***"
            
#            print "---"

#        for (k,v) in Dict_features.items():
#            print k+": " + str(v)

#        print len(Dict_features)

                                    
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

    def computeGramsExplicit(self, g_it, approx =True, precomputed =None):
        if precomputed is None:
            precomputed = self.transform_serial_explicit(g_it, approximated = approx)

        return [matrix(np.array(p.dot(p.T).todense().tolist())) for p in precomputed]

    def computeKernelMatricesTrain(self, Graphs):
        return self.computeGramsExplicit(Graphs)   

