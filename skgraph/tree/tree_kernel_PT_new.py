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
along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
"""
from copy import deepcopy
import math
from skgraph import graph

import numpy as np
import sys

#######################################################################################

class PTKernel(): #Kernel
    def __init__(self,l,m,normalize=True,hashsep="#"):
        self.l = float(l)
        self.mu = float(m)
        self.hashsep = hashsep
        self.normalize = normalize
        self.cache = Cache()    

    def preProcess(self,T):
        if hasattr(T,'kernelptrepr'): #already preprocessed
            return 

        #ordinare se l'albero non e' gia' ordered
        if not 'childrenOrder' in T.nodes(data=True)[0][1]:
            print 'Error: children are not ordered!'
            
       
        #a.root.setHashSubtreeIdentifier(self.hashsep)
        if not 'kernelptrepr' in T.graph:
            T.graph['kernelptrepr'] = LabelSubtreeList(T)
            T.graph['kernelptrepr'].sort()
        if not 'norm' in T.graph:     
            if self.normalize:
                T.graph['norm'] = 1.0
                b = deepcopy(T)
                T.graph['norm'] = math.sqrt(self.evaluate(T,b))

    def DeltaSk(self,T1, a,T2, b,nca, ncb):
        DPS = [[0 for i in xrange(ncb+1)] for j in xrange(nca+1)]
        DP = [[0 for i in xrange(ncb+1)] for j in xrange(nca+1)]
        kmat = [0]*(nca+1)
        
        children1=T1.node[a]['childrenOrder']
        children2=T2.node[b]['childrenOrder']
        for i in xrange(1,nca+1):
            for j in xrange(1,ncb+1):
                if T1.node[children1[i-1]]['label'] == T2.node[children2[j-1]]['label']:
                    DPS[i][j] = self.CPT(T1,children1[i-1],T2,children2[j-1])
                    kmat[0] += DPS[i][j]
                else:
                    DPS[i][j] = 0
        for s in xrange(1,min(nca,ncb)):
            for i in xrange(nca+1): 
                DP[i][s-1] = 0
            for j in xrange(ncb+1): 
                DP[s-1][j] = 0
            for i in xrange(s,nca+1):
                for j in xrange(s,ncb+1):
                    DP[i][j] = DPS[i][j] + self.mu*DP[i-1][j] + self.mu*DP[i][j-1] - self.mu**2*DP[i-1][j-1]
                    if T1.node[children1[i-1]]['label'] == T2.node[children2[j-1]]['label']:
                        DPS[i][j] = self.CPT(T1,children1[i-1],T2,children2[j-1])*DP[i-1][j-1]
                        kmat[s] += DPS[i][j]
        return sum(kmat)
    
    def CPT(self,T1,c,T2,d):
        tmpkey = str(c) + "#" + str(d) 
        if self.cache.exists(tmpkey):
            return self.cache.read(tmpkey)
        else:
            if len(T1.node[c]['childrenOrder'])==0 or len(T2.node[d]['childrenOrder'])==0:
                prod = self.l*self.mu**2
            else:
                prod = self.l*(self.mu**2+self.DeltaSk(T1,c,T2, d,len(T1.node[c]['childrenOrder']),len(T2.node[d]['childrenOrder'])))
            
            
            self.cache.insert(tmpkey, prod)
            return prod     


    def evaluate(self,a,b):
        self.cache.removeAll()
#        self.preProcess(a)
#        self.preProcess(b)
        la,lb = (a.graph['kernelptrepr'],b.graph['kernelptrepr'])
        i,j,k,toti,totj = (0,0,0.0,len(la),len(lb))
        while i < toti and j < totj:
            if la.getNodeLabel(i) == lb.getNodeLabel(j):
                ci,cj=(i,j)
                while i < toti and la.getNodeLabel(i) == la.getNodeLabel(ci):
                    j = cj
                    while j < totj and lb.getNodeLabel(j) == lb.getNodeLabel(cj):
                        k += self.CPT(a,la.getTree(i),b,lb.getTree(j))
                        j += 1
                    i += 1
            elif la.getNodeLabel(i) <= lb.getNodeLabel(j):
                i += 1
            else:
                j += 1
        if self.normalize:
              k = k/(a.graph['norm']*b.graph['norm'])
        return k
        
    def kernel(self, a, b):
        """compute the tree kernel on the trees a and b"""
#        if not isinstance(a, tree.Tree):
#            print "ERROR: first parameter has to be a Tree Object"
#            return ""
#        if not isinstance(b, tree.Tree):
#            print "ERROR: second parameter has to be a Tree Object"
#            return ""
        self.preProcess(a)
        self.preProcess(b)
        return self.evaluate(a,b)
        
    def __str__(self):
        return "Partial Tree Kernel, with lambda=" + self.l + " and mu=" + self.mu
    
    def computeKernelMatrix(self,Graphs):
        #TODO
        print "Computing gram matrix"
        import numpy as np
        import sys
        Gram = np.empty(shape=(len(Graphs),len(Graphs)))
        progress=0
        for  i in xrange(0,len(Graphs)):
            for  j in xrange(i,len(Graphs)):
                #print "COMPUTING GRAPHS",i,j
                progress+=1
                Gram[i][j]=self.kernel(Graphs[i],Graphs[j])
                Gram[j][i]=Gram[i][j]
                if progress % 1000 ==0:
                    print "k",
                    sys.stdout.flush()
                elif progress % 100 ==0:
                    print ".",
                    sys.stdout.flush()
    
        return Gram
        


def setHashSubtreeIdentifier(T, nodeID, sep='|',labels=True):
        """
        The method computes an identifier of the node based on
        1) the label of the node self
        2) the hash values of the children of self
        For each visited node the hash value is stored into the attribute subtreeId
        The label and the identifiers of the children nodes are separated by the char 'sep'
        """
        #print "labels",labels

        if 'subtreeID' in T.node[nodeID]:
            return T.node[nodeID]['subtreeID']
        if labels:
            stri = str(T.node[nodeID]['label'])
        else:
            stri = str(T.out_degree(nodeID))
        if stri.find(sep) != -1:
            print "ERROR: identifier " + sep + "used in label. Please set it with setHashSep(newsep)"
        for c in T.node[nodeID]['childrenOrder']:#T.successors(nodeID):
            stri += sep + setHashSubtreeIdentifier(T,c,sep)
        T.node[nodeID]['subtreeID'] = str(hash(stri))
        return T.node[nodeID]['subtreeID']


def computeSubtreeIDSubtreeSizeList(self):
        #compute a list of pairs (subtree-hash-identifiers, subtree-size)
        if not self:
            return
        p = [(self.subtreeId, self.stsize)]
        for c in self.chs:
            p.extend(c.computeSubtreeIDSubtreeSizeList())
        return p
        
class LabelSubtreeList():
    """The class builds the data structure used by the route kernel"""
    def __init__(self, T):
        self.labelList = labelList(T,T.graph['root'])

    def getNodeLabel(self,i):
        return self.labelList[i][0]

    def getTree(self,i):
        return self.labelList[i][1]

    def getIndex(self,i):
        return self.labelList[i][2]
    
    def sort(self):
        self.labelList.sort(cmp = lambda x, y: cmp(x[0], y[0]))

    def __len__(self):
        return len(self.labelList)
        
        
#####################classe TREE 
# prima di bigDAG
def labelList(T,a): 
    """
    The method returns, for all descendants of the node, a triplet composed by 
    the label, a reference to the node, the index of the node w.r.t its siblings.
    Index is ignored and set to zero        
    """
    #p = [(self.val,self, self.ind)]
    p = [(T.node[a]['label'],a, 0)]

    for c in T.node[a]['childrenOrder']:
        p.extend(labelList(T,c))
    return p        
        
#def labelListbigDAG(T,roots): 
#    """
#    The method returns, for all descendants of the node, a triplet composed by 
#    the label, a reference to the node, the index of the node w.r.t its siblings.
#    Index is ignored and set to zero        
#    """
#    #p = [(self.val,self, self.ind)]
#    p=[]
#    for a in roots:
#        p.extend([T.node[a]['label'],a, 0])
#    
#        for c in T.node[a]['childrenOrder']:
#            p.extend(labelList(T,[c]))
#    return p

    def getLeafLabelList(self):
        """
        The method returns the list of node labels for all leaf nodes which are descendants of self.
        """
        if not self: return []
        if self.isLeaf():
            return [self.val]
        else:
            p = []
            for c in self.chs:
                p += c.getLeafLabelList()
            return p
            
class Cache():
    """
    An extremely simple cache 
    """

    def __init__(self):
        self.cache = {} 

    def exists(self,key):
        return key in self.cache

    def existsPair(self,keya,keyb):
        if keya < keyb:
            tmpkey = str(keya) + "#" + str(keyb)
        else:
            tmpkey = str(keyb) + "#" + str(keya) 
        return tmpkey in self.cache

    def insert(self,key,value):
        self.cache[key] = value

    def insertPairIfNew(self,keya,keyb):
        if keya < keyb:
            tmpkey = str(keya) + "#" + str(keyb)
        else:
            tmpkey = str(keyb) + "#" + str(keya) 
        if not tmpkey in self.cache:
            self.insert(tmpkey)

    def remove(self,key):
        del self.cache[key]

    def removeAll(self):
        self.cache = {}

    def read(self,key):
        return self.cache[key]