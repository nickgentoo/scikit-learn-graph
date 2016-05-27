__author__ = "Nicolo' Navarin"
__date__ = "13/feb/2015"


from graphKernel import GraphKernel
from scipy.sparse import csr_matrix
from ..graph import graph
from ..graph.GraphTools import generateDAG
import numpy as np
import sys
#from dependencies.pythontk import tree_kernels_new
from ..tree import tree_kernels_STonlyroot_FeatureVector_ApproxRBF
from sklearn.kernel_approximation import RBFSampler, Nystroem

from multiprocessing import Pool
from functools import partial

#suxiliary functions for parallelization
def calculate_kernel(Graph1,GraphKernel,Graph2):
    return GraphKernel.kernelFunction(Graph1,Graph2)
def calculate_kernelFeatureVector(Graph1,GraphKernel,Graph2):
    return GraphKernel.kernelFunctionFeatureVectors(Graph1,Graph2)
    
def calculate_kernelBigDAG(Graph1,GraphKernel,Graph2):
    return GraphKernel.kernelFunctionBigDAG(Graph1,Graph2)

class ODDCLApproxRBFGraphKernel(GraphKernel):
    """
    Class that implements the ODDKernel with ST kernel
    """
    
    def __init__(self, r = 3, l = 1, normalization = True, hashsz=32,n_comp=1000):
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
        self.n_comp=n_comp
        self.Lambda=l
        self.max_radius=r
        self.normalization=normalization
        self.__hash_size=hashsz 
        self.__bitmask = pow(2, hashsz) - 1
        self.__startsymbol='!' #special symbols used in encoding
        self.__conjsymbol='#'
        self.__endsymbol='?'
        self.treekernelfunction=tree_kernels_STonlyroot_FeatureVector_ApproxRBF.STKernel(self.Lambda,order='gaussian')

    def generateGraphFeatureMap(self,Graph,radius):
        FeatureMap={}
        for v1 in Graph.nodes():
            for radius in xrange(radius+1):

                (DAG,maxLevel)=generateDAG(Graph, v1,radius)
                DAG.graph['root']=v1
                #print DAG.nodes()
                self.treekernelfunction._addToFeatureMap(FeatureMap,DAG)  
                
        FeatureMap1=self.treekernelfunction._computeSum(FeatureMap)
        return FeatureMap1
    @staticmethod
#    def __shrink(col):
#        """
#        Private method that compresses the column index vector of some csr_matrix
#        @type col: numpy array
#        @param col: array that represents the column indexes of a csr_matrix
#        
#        @rtype: list
#        @return: list of the compressed columns
#        """
#        minimums=np.sort(np.unique(col))
#        for minindex in range(len(minimums)):
#            currentmin=minimums[minindex]
#            col[np.where(col==currentmin)]=minindex
#        return col.tolist()

    def kernelFunction(self,Graph1, Graph2):#TODO inutile
        #calculate the kernel between two FeatureMaps
        FM1=self.generateGraphFeatureMap(Graph1,self.max_radius)
        FM2=self.generateGraphFeatureMap(Graph2,self.max_radius)

        return self._kernelFunctionFeatureVectors(FM1,FM2)
        

    def _kernelFunctionFeatureVectors(self,FeatureVector1, FeatureVector2):#TODO inutile
        #calculate the kernel between two FeatureMaps
        kernel=0
        kernel += self.treekernelfunction.kernelFeatureMap(FeatureVector1,FeatureVector2)
        return kernel   
        
   
    def computeKernelMatrix(self,Graphs):
        print "Computing gram matrix"
        #self.treekernelfunction=tree_kernels_STonlyroot_FeatureVector.STKernel(self.Lambda, labels=self.labels,veclabels=self.veclabels,order=self.order)        
        #Preprocessing step: approximation of RBF with explicit features.
        #Add a field to every node "veclabel_explicit_rbf"
        labels=set()
        for g in Graphs:
            for _,d in g.nodes(data=True):
                #print d
                labels.add(tuple(d['veclabel']))
        #print len(labels)
        labels_list=[list(l) for l in labels]
##                labels=set()
#        labels_list=[]
#        for g in Graphs:
#            for _,d in g.nodes(data=True):
#                #print d
#                labels_list.append(list(d['veclabel']))
        #print len(labels)
        #labels_list=[list(l) for l in labels]
        print "Size of labels matrix:",len(labels_list),len(labels_list[0])
        feature_map_fourier = RBFSampler(gamma=(1.0/len(labels_list[0])), random_state=1,n_components=self.n_comp)
        #feature_map_fourier = Nystroem(gamma=(1.0/len(labels_list[0])), random_state=1,n_components=250)

        feature_map_fourier.fit(labels_list)
        for g in Graphs:
            for n,d in g.nodes(data=True):
               g.node[n]['veclabel_rbf']=feature_map_fourier.transform(d['veclabel'])[0] #.tolist()
        print "RBF approximation finished."
        #print Graphs[0].node[1]['veclabel_rbf']

        
        Gram = np.empty(shape=(len(Graphs),len(Graphs)))
        progress=0
        FeatureMaps=[]
        for  i in xrange(0,len(Graphs)):
            FeatureMaps.append(self.generateGraphFeatureMap(Graphs[i],self.max_radius))
          
        print "FeatureVectors calculated"        
        for  i in xrange(0,len(Graphs)):
            for  j in xrange(i,len(Graphs)):
                #print i,j
                progress+=1
                Gram[i][j]=self._kernelFunctionFeatureVectors(FeatureMaps[i],FeatureMaps[j])
                Gram[j][i]=Gram[i][j]
                if progress % 1000 ==0:
                    print "k",
                    sys.stdout.flush()
                elif progress % 100 ==0:
                    print ".",
                    sys.stdout.flush()

        return Gram   
   
   
        
#    def computeKernelMatrixFeatureVectorParallel(self,Graphs,njobs=-1):
#        print "Computing gram matrix"
#        if self.tree_kernel=="STonlyroot":
#             self.treekernelfunction=tree_kernels_STonlyroot_FeatureVector.STKernel(self.Lambda, labels=self.labels,veclabels=self.veclabels,order=self.order)        
#        else:
#            print "ERROR: tree kernel unknown"
#        Gram = np.empty(shape=(len(Graphs),len(Graphs)))
#        progress=0
#        if njobs == -1:
#            njobs = None
#
#        pool = Pool(njobs)
#                
#        
#        FeatureVectors=[]
#        for  i in xrange(0,len(Graphs)):
#            FeatureVectors.append(self.generateGraphFeatureMap(Graphs[i],self.max_radius))
#
#        print "FeatureVectors calculated"              
#        for  i in xrange(0,len(Graphs)):
#            partial_calculate_kernel = partial(calculate_kernelFeatureVector, GraphKernel=self,Graph2=FeatureVectors[i])
#
#                
#            jth_column=pool.map(partial_calculate_kernel,[FeatureVectors[j] for j in range(i,len(Graphs))])
#            for  j in xrange(i,len(Graphs)):
#                #print i,j-i
#                progress+=1
#                Gram[i][j]=jth_column[j-i]
#                Gram[j][i]=Gram[i][j]
#                if progress % 1000 ==0:
#                    print "k",
#                    sys.stdout.flush()
#                elif progress % 100 ==0:
#                    print ".",
#                    sys.stdout.flush()
#
#        return Gram
