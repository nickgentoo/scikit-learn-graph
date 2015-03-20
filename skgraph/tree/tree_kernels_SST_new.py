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
from skgraph import graph
from scipy.spatial.distance import pdist
import numpy as np
import sys
from numpy.linalg import norm
def gaussianKernel(X,Y,beta):
    Z=[X,Y]
    return np.exp(-beta * pdist(Z, 'sqeuclidean'))
def linearKernel(X,Y):
    Xnum=np.array(X)
    Ynum=np.array(Y)    
    Xnum=Xnum/norm(Xnum)
    Ynum=Ynum/norm(Ynum) 
    return np.dot(Xnum,Ynum)


class SSTKernel:
    def __init__(self,l,normalize=True,hashsep="#"):
        self.l = float(l)
        self.hashsep = hashsep
        self.cache = Cache()


    def preProcess(self,T):
        if 'kernelsstrepr' in T.graph:
            return 
        #a['hashsep']=self.hashsep
        #ordinare se l'albero non e' gia' ordered
        if not 'childrenOrder' in T.nodes(data=True)[0][1]:
            print 'ERROR: children are not ordered!'
        #graph.setHashSubtreeIdentifier(T,T.graph['root'],self.hashsep)

        T.graph['kernelsstrepr']=ProdSubtreeList(T,T.graph['root'],labels=self.labels)
        T.graph['kernelsstrepr'].sort()

    def evaluate(self,a,b):
        #self.preProcess(a)
        #self.preProcess(b)
            
        pa,pb=(a.graph['kernelsstrepr'], b.graph['kernelsstrepr'])
#        print "PA"
#        for i in range(len(pa)):
#            print pa.getProduction(i),
#        print "PB"
#        for j in range(len(pb)):
#            print pb.getProduction(j),
        self.cache.removeAll()
        i,j,k,toti,totj = (0,0,0,len(pa),len(pb))
        while i < toti and j < totj:
            if a.node[pa.getTree(i)]['label']==b.node[pb.getTree(j)]['label']:
                ci,cj=(i,j)
                while i < toti and a.node[pa.getTree(i)]['label']==a.node[pa.getTree(ci)]['label']:
                    j = cj
                    while j < totj and b.node[pb.getTree(j)]['label']==b.node[pb.getTree(cj)]['label']:
                        k += self.CSST(a, pa.getTree(i),b, pb.getTree(j))
                        j += 1
                    i += 1
            elif len(pa.getProduction(i))<len(pb.getProduction(j)) or (len(pa.getProduction(i))==len(pb.getProduction(j)) and pa.getProduction(i) < pb.getProduction(j)):
                i += 1
            else:
                j += 1
        return k

    def kernel(self, a, b):
        """compute the tree kernel on the trees a and b"""

        self.preProcess(a)
        self.preProcess(b)
        return self.evaluate(a,b) 

          

    def CSST(self,G1,c,G2,d):
        #ASSUME THAT C AND D HAVE THE SAME LABEL
        #c and d are node indices
        tmpkey = str(c) + "#" + str(d)
        if self.cache.exists(tmpkey):
#            print "existing key value ",self.cache.read(tmpkey)
            return float(self.cache.read(tmpkey))
        else:
            prod = self.l
#            print "roots",G1.node[c]['label'],",",G2.node[d]['label']
            #if 'veclabel' in G1.nodes(data=True)[0][1]:
            if self.veclabels:
                    prod*=gaussianKernel(G1.node[c]['veclabel'],G2.node[d]['veclabel'],1.0/len(G2.node[d]['veclabel']))
            
            children1=G1.node[c]['childrenOrder']
            children2=G2.node[d]['childrenOrder']
            nc = G1.out_degree(c)
            if nc==G2.out_degree(d) and nc>0:
                if getProduction(G1,c) == getProduction(G2,d):
#                    print "productions",graph.getProduction(G1,c),graph.getProduction(G2,d)
                    for ci in range(nc):
#                       print "child",ci
#                       print G1.node[children1[ci]]['label'],",",G2.node[children2[ci]]['label']
                       prod *= (1 +self.CSST(G1,children1[ci],G2,children2[ci]))
                       #print "SST partial", SST_partial 
                    #BUG
            self.cache.insert(tmpkey, prod)
        return float(prod)

#originale + label vettoriali
#    def CSST(self,G1,c,G2,d):
#        #ASSUME THAT C AND D HAVE THE SAME LABEL
#        #c and d are node indices
#        tmpkey = str(c) + "#" + str(d)
#        if self.cache.exists(tmpkey):
#            return float(self.cache.read(tmpkey))
#        else:
#            prod = self.l
#            #print G1.node[c]
#            #if 'veclabel' in G1.nodes(data=True)[0][1]:
#            if self.veclabels:
#                if self.nodekernel=="gaussian":
#                    prod*=gaussianKernel(G1.node[c]['veclabel'],G2.node[d]['veclabel'],1.0/len(G2.node[d]['veclabel']))#self.beta)
#                elif self.nodekernel=="linear":
#                    prod*=linearKernel(G1.node[c]['veclabel'],G2.node[d]['veclabel'])#self.beta)
#            #print "kernel etichette", gaussianKernel(G1.node[c]['veclabel'],G2.node[d]['veclabel'],self.beta)
#            #print "no vector label detected"            
#            #TODO qui va aggiunto il kenrel tra label continue
#            children1=G1.node[c]['childrenOrder']
#            children2=G2.node[d]['childrenOrder']
#            nc = G1.out_degree(c)
#            if nc==G2.out_degree(d):
#                for ci in range(nc):
#                    if graph.getProduction(G1,children1[ci]) == graph.getProduction(G2,children2[ci]):
#                         self.CSST(G1,children1[ci],G2,children2[ci]))
#                        #TODO qui non credo
#                    else:
#                        #print G1.node[children1[ci]]['subtreeID']
#                        cid, did = (children1[ci],children2[ci])
#                        self.cache.insert(str(cid) +"#"+ str(did), 0)
#
#            self.cache.insert(tmpkey, prod)
#        return float(prod)

#    def evaluate(self,a,b):
#        pa,pb=(a.kernelsstrepr, b.kernelsstrepr)
#        self.cache.removeAll()
#        i,j,k,toti,totj = (0,0,0,len(pa),len(pb))
#        while i < toti and j < totj:
#            if pa.getProduction(i) == pb.getProduction(j):
#                ci,cj=(i,j)
#                while i < toti and pa.getProduction(i)==pa.getProduction(ci):
#                    j = cj
#                    while j < totj and pb.getProduction(j)==pb.getProduction(cj):
#                        k += self.CSST(pa.getTree(i),pb.getTree(j))
#                        j += 1
#                    i += 1
#            elif len(pa.getProduction(i))<len(pb.getProduction(j)) or (len(pa.getProduction(i))==len(pb.getProduction(j)) and pa.getProduction(i) < pb.getProduction(j)):
#                i += 1
#            else:
#                j += 1
#        return k

    def __str__(self):
        return "Subset Tree Kernel, with lambda=" + self.l
        
    def computeKernelMatrix(self,Graphs):
        #TODO
        print "Computing gram matrix"
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

#class SSTKernelOrdered(SSTKernel):
#
#    def preProcess(self,T):
#        if 'kernelsstrepr' in T.graph:
#            return 
#        #a['hashsep']=self.hashsep
#        #ordinare se l'albero non e' gia' ordered
#        #for n in T.nodes():
#        #    T.node[n]['childrenOrder']=T.successors(n)
#        #graph.setHashSubtreeIdentifier(T,T.graph['root'],self.hashsep)
#
#        T.graph['kernelsstrepr']=graph.ProdSubtreeList(T,T.graph['root'])
#        T.graph['kernelsstrepr'].sort()

class SSTAllMatchingKernel(SSTKernel):

    def CSST(self,G1,c,G2,d):
        #ASSUME THAT C AND D HAVE THE SAME LABEL
        #c and d are node indices
        tmpkey = str(c) + "#" + str(d)
        if self.cache.exists(tmpkey):
#            print "existing key value ",self.cache.read(tmpkey)
            return float(self.cache.read(tmpkey))
        else:
            prod = self.l
#            print "roots",G1.node[c]['label'],",",G2.node[d]['label']
            #if 'veclabel' in G1.nodes(data=True)[0][1]:
            if self.veclabels:
                if self.nodekernel=="gaussian":
                    prod*=gaussianKernel(G1.node[c]['veclabel'],G2.node[d]['veclabel'],1.0/len(G2.node[d]['veclabel']))#self.beta)
                elif self.nodekernel=="linear":
                    prod*=linearKernel(G1.node[c]['veclabel'],G2.node[d]['veclabel'])#self.beta)
            #print "kernel etichette", gaussianKernel(G1.node[c]['veclabel'],G2.node[d]['veclabel'],self.beta)
            #print "no vector label detected"            
            #TODO qui va aggiunto il kenrel tra label continue
            #primo controllo tra produzioni

            #TODO qui non credo
            
            children1=G1.node[c]['childrenOrder']
            children2=G2.node[d]['childrenOrder']
            nc = G1.out_degree(c)
            if nc==G2.out_degree(d) and nc>0:
                if getProduction(G1,c) == getProduction(G2,d):
#                    print "productions",graph.getProduction(G1,c),graph.getProduction(G2,d)
                    #TEST
                    i,j,k,toti,totj = (0,0,0,nc,nc)
                    if G1.node[children1[ci]]['label']==G2.node[children2[cj]]['label']:
                        ci,cj=(i,j)
                        while i < toti and G1.node[children1[i]]['label']==G1.node[children1[ci]]['label']:
                            j = cj
                            while j < totj and G2.node[children2[j]]['label']==G2.node[children2[cj]]['label']:
                                k += self.CSST(G1, children1[i],G2, children2[j])
                                j += 1
                                i += 1
                    elif len(pa.getProduction(i))<len(pb.getProduction(j)) or (len(pa.getProduction(i))==len(pb.getProduction(j)) and pa.getProduction(i) < pb.getProduction(j)):
                        i += 1
                    else:
                        j += 1
                    #fine test
                    
                    SST_partial=1               
                    for ci in range(nc):
                        for cj in range(nc):
                            if G1.node[children1[ci]]['label']==G2.node[children2[cj]]['label']:
#                       print "child",ci
#                       print G1.node[children1[ci]]['label'],",",G2.node[children2[ci]]['label']
                        #TODO se etichette diverse non deve andare a 0
                                SST_partial*=self.CSST(G1,children1[ci],G2,children2[cj])
                       #print "SST partial", SST_partial 
                    prod *= (1 + SST_partial )            
            #print "after first loop", prod
#            if nc==G2.out_degree(d):
#                for ci in range(nc):
#                    #if G1.node[children1[ci]]['label']==G2.node[children2[ci]]['label']:
#                    if graph.getProduction(G1,children1[ci]) == graph.getProduction(G2,children2[ci]):
#                        prod *= (1 + self.CSST(G1,children1[ci],G2,children2[ci]))
#                        #TODO qui non credo
#                    else:
#                        #print G1.node[children1[ci]]['subtreeID']
#                        cid, did = (children1[ci],children2[ci])
#                        #self.cache.insert(str(cid) +"#"+ str(did), 0)

            #print "total", prod            
            self.cache.insert(tmpkey, prod)
        return float(prod)
        
    def CSSTProductions(self,G1,c,G2,d):
        #c and d are node indices
        tmpkey = str(c) + "#" + str(d)
        if self.cache.exists(tmpkey):
            return float(self.cache.read(tmpkey))
        else:
            prod = self.l
            #print G1.node[c]
            #if 'veclabel' in G1.nodes(data=True)[0][1]:
            if self.veclabels:
                if self.nodekernel=="gaussian":
                    prod*=gaussianKernel(G1.node[c]['veclabel'],G2.node[d]['veclabel'],1.0/len(G2.node[d]['veclabel']))#self.beta)
                elif self.nodekernel=="linear":
                    prod*=linearKernel(G1.node[c]['veclabel'],G2.node[d]['veclabel'])#self.beta)

                #print "kernel etichette", gaussianKernel(G1.node[c]['veclabel'],G2.node[d]['veclabel'],self.beta)
            #print "no vector label detected"            
            #TODO qui va aggiunto il kenrel tra label continue
            children1=G1.node[c]['childrenOrder']
            children2=G2.node[d]['childrenOrder']
            nc = G1.out_degree(c)
            if nc==G2.out_degree(d):
                for ci in range(nc):
                    for cj in range(nc):
                        if getProduction(G1,children1[ci]) == getProduction(G2,children2[cj]):
                            prod *= (1 + self.CSST(G1,children1[ci],G2,children2[cj]))
                            #TODO qui non credo
                        else:
                            #print G1.node[children1[ci]]['subtreeID']
                            cid, did = (children1[ci],children2[cj])
                            self.cache.insert(str(cid) +"#"+ str(did), 0)
    
            self.cache.insert(tmpkey, prod)
        return float(prod)
    
    #    def evaluate(self,a,b):
    #        pa,pb=(a.kernelsstrepr, b.kernelsstrepr)
    #        self.cache.removeAll()
    #        i,j,k,toti,totj = (0,0,0,len(pa),len(pb))
    #        while i < toti and j < totj:
    #            if pa.getProduction(i) == pb.getProduction(j):
    #                ci,cj=(i,j)
    #                while i < toti and pa.getProduction(i)==pa.getProduction(ci):
    #                    j = cj
    #                    while j < totj and pb.getProduction(j)==pb.getProduction(cj):
    #                        k += self.CSST(pa.getTree(i),pb.getTree(j))
    #                        j += 1
    #                    i += 1
    #            elif len(pa.getProduction(i))<len(pb.getProduction(j)) or (len(pa.getProduction(i))==len(pb.getProduction(j)) and pa.getProduction(i) < pb.getProduction(j)):
    #                i += 1
    #            else:
    #                j += 1
    #        return k
    
#############################
## GRAPH FUNCTIONS
def setOrder(T, nodeID, sep='|',labels=True,veclabels=True,order="gaussian"):
    if veclabels:
        return setOrderVeclabels(T, nodeID,sep,labels,order)
    else:
       return setOrderNoVeclabels(T, nodeID,sep,labels)


def setOrderVeclabels(T, nodeID, sep,labels,order):
        """
        The method computes an identifier of the node based on
        1) the label of the node self
        2) the hash values of the children of self
        For each visited node the hash value is stored into the attribute subtreeId
        The label and the identifiers of the children nodes are separated by the char 'sep'
        """
        #print "labels",labels
        #print "sep",sep

        if 'orderString' in T.node[nodeID]:
            return T.node[nodeID]['orderString']
        if labels==True:
            stri = str(T.node[nodeID]['label'])
        else:
            stri = str(T.out_degree(nodeID))
        #order according kernel between veclabels and the one of the father
 
            
        #print str(T.node[nodeID]['veclabel'])
        #print stri
        if stri.find(sep) != -1:
            print "ERROR: identifier " + sep + "used in label. Please set it with setHashSep(newsep)"
        succ_labels=[]
        if len(T.successors(nodeID))>0:
            stri +=sep+str(len(T.successors(nodeID)))
        for c in T.successors(nodeID):
            if order=='gaussian':
                dist=gaussianKernel(T.node[nodeID]['veclabel'],T.node[c]['veclabel'],1.0/len(T.node[c]['veclabel']))#self.beta)
            elif order=='norm':
                dist=norm(T.node[nodeID]['veclabel'])
            else:
                print "no ordering specified"
            tup=([setOrderVeclabels(T,c,sep,labels,order),dist],c)
            succ_labels.append(tup)
        #print "before sorting",succ_labels
        succ_labels.sort(key=lambda x:(x[0][0],x[0][1]))#cmp = lambda x, y: cmp(x[0], y[0])
        #print "after sorting",succ_labels
        children=[]
        for l in  succ_labels:
            stri += sep + str(l[0][0])
            children.append(l[1])
        T.node[nodeID]['orderString'] = stri
        #print "order string", stri
        T.node[nodeID]['childrenOrder']= children
        #print "order children", children

        #debug
        #print T.node[nodeID]['orderString']
        #print T.node[nodeID]['childrenOrder']
        return T.node[nodeID]['orderString']

def setOrderNoVeclabels(T, nodeID, sep,labels,order):
        """
        The method computes an identifier of the node based on
        1) the label of the node self
        2) the hash values of the children of self
        For each visited node the hash value is stored into the attribute subtreeId
        The label and the identifiers of the children nodes are separated by the char 'sep'
        """
        #print "labels",labels
        #print "sep",sep

        if 'orderString' in T.node[nodeID]:
            return T.node[nodeID]['orderString']
        if labels==True:
            stri = str(T.node[nodeID]['label'])
        else:
            stri = str(T.out_degree(nodeID))
        #append veclabels if available
        #print str(T.node[nodeID]['veclabel'])
        #print stri
        if stri.find(sep) != -1:
            print "ERROR: identifier " + sep + "used in label. Please set it with setHashSep(newsep)"
        succ_labels=[]
        if len(T.successors(nodeID))>0:
            stri +=sep+str(len(T.successors(nodeID)))
        for c in T.successors(nodeID):
            tup=(setOrderNoVeclabels(T,c,sep,labels,order),c)
            succ_labels.append(tup)
            succ_labels.sort(cmp = lambda x, y: cmp(x[0], y[0]))
        children=[]
        for l in  succ_labels:
            stri += sep + str(l[0])
            children.append(l[1])
        T.node[nodeID]['orderString'] = stri
        T.node[nodeID]['childrenOrder']= children
        #debug
        #print T.node[nodeID]['orderString']
        #print T.node[nodeID]['childrenOrder']
        return T.node[nodeID]['orderString']


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


class ProdSubtreeList():
    def __init__(self,T, root,labels=True):
        self.labels=labels
        self.productionlist = self.productionlist(T,root)

    def getProduction(self,i):
        return self.productionlist[i][0]

    def getTree(self,i):
        return self.productionlist[i][1]

    def sort(self):
        #erano invertiti quando confrontavo produzioni
        self.productionlist.sort(cmp = lambda x, y: cmp(len(x[0]), len(y[0])))
        self.productionlist.sort(cmp = lambda x, y: cmp(x[0], y[0]))

#ORIGINAL SORT FOR SST CONSIDERING PRODUCTINS
#    def sort(self):
#        self.productionlist.sort(cmp = lambda x, y: cmp(x[0], y[0]))
#        self.productionlist.sort(cmp = lambda x, y: cmp(len(x[0]), len(y[0])))


    def __len__(self):
        return len(self.productionlist)

    def compareprods(x,y):
        if x[0]==y[0]: 
            return cmp(len(x[0]),len(y[0]))
        else:
            return cmp(x[0],y[0])


    def productionlist(self,G,nodeID):
            p = [(getProduction(G,nodeID,self.labels),nodeID)]
            for c in G.successors(nodeID):
                p.extend(self.productionlist(G,c))
            return p

def getProduction(G, nodeID,labels=True): 
        """
        The method returns a string representing the label of the current node (self) concatenated with the labels of its children
        The format of the string is the following: l_v(l_ch1,l_ch2,...,l_chn)
        where l_v is the label of self and l_chi is the label of the i-th child. 
        For example the string representing a subtree composed by a node labelled with A and two children labelled as B and C, 
        is represented as A(B,C)
        The empty string is returned in case the node is not a TreeNode object properly initialized. 
        """
        if 'production' in G.node[nodeID]:
            return G.node[nodeID]['production']

        #print nodeID
        #print "labels",labels
        if labels:
            G.node[nodeID]['production'] = G.node[nodeID]['label'] + "(" + ','.join([G.node[childID]['label'] for childID in G.node[nodeID]['childrenOrder']]) + ")"
        else:
            #TODO outdegree
            G.node[nodeID]['production'] =str(G.out_degree(nodeID)) + "(" + ','.join([str(G.out_degree(childID)) for childID in G.node[nodeID]['childrenOrder']]) + ")"

        #print G.node[nodeID]['production']        
        return G.node[nodeID]['production'] 