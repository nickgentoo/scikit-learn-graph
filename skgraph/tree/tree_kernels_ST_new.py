#from .... import graph
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
    

    
class STKernel():
    def __init__(self,l,hashsep="#",labels=True,veclabels=False,order="gaussian"):
        self.order=order      
        self.l = float(l)
        self.hashsep = hashsep
        self.labels=labels
        self.veclabels=veclabels
        self.cache = Cache()


    def preProcess(self,a):
        if hasattr(a,'kernelstrepr'): #already preprocessed
            return 
            
        if not 'childrenOrder' in a.nodes(data=True)[0][1]:
            setOrder(a,a.graph['root'],sep=self.hashsep,labels=self.labels,veclabels=self.veclabels,order=self.order)
     
        #if not hasattr(a.root, 'stsize'):
        #    getSubtreeSize(T,T.graph['root'])
        #a.root.setHashSubtreeIdentifier(self.hashsep)
        a.graph['kernelstrepr'] = SubtreeIDSubtreeSizeList(a,a.graph['root'])
        a.graph['kernelstrepr'] .sort()

    def computeSubtreeVecLabelKernels(self,G1,a,G2,b):
        tmpkey = str(a) + "#" + str(b)
        if self.cache.exists(tmpkey):
#            print "existing key value ",self.cache.read(tmpkey)
            return float(self.cache.read(tmpkey))
        else:
            k=gaussianKernel(G1.node[a]['veclabel'],G2.node[b]['veclabel'],1.0/len(G2.node[b]['veclabel']))#self.beta)
            children1=G1.node[a]['childrenOrder']
            children2=G2.node[b]['childrenOrder']
            nc = G1.out_degree(a)
            for i in range(nc):
                k*=self.computeSubtreeVecLabelKernels(G1,children1[i],G2,children2[i])
            self.cache.insert(tmpkey, k)
            return k
    
    def evaluate(self,a,b):
        #Assumes ha and hb are ordered list of pairs (subtreeid, subtreesize) 
        #a.kernelreprst,b.kernelreprst are checked or created in preProcess()
        ha, hb = (a.graph['kernelstrepr'], b.graph['kernelstrepr'])
        self.cache.removeAll()

#        print "HA"
#        for i in range(len(ha)):
#           print ha.getSubtreeID(i),
#        print "HB"
#        for i in range(len(hb)):
#           print hb.getSubtreeID(i),
        i,j,k,toti,totj = (0,0,0,len(ha), len(hb))
        while i < toti and j < totj:
            if ha.getSubtreeID(i) == hb.getSubtreeID(j):
                ci,cj=(i,j)
                #ORIGINAL CODE
#                while i < toti and ha.getSubtreeID(i)==ha.getSubtreeID(ci):
#                    i += 1
#                while j < totj and hb.getSubtreeID(j)==hb.getSubtreeID(cj):
#                    j += 1
#                k += (i-ci)*(j-cj)*(self.l**ha.getSubtreeSize(ci))
                
                while i < toti and ha.getSubtreeID(i)==ha.getSubtreeID(ci):
                    j = cj
                    while j < totj and hb.getSubtreeID(j)==hb.getSubtreeID(cj):
                        #node_kernel=1.0                        
                        if self.veclabels:
                            #node_kernel=gaussianKernel(a.node[ha.getNodeIndex(i)]['veclabel'],b.node[hb.getNodeIndex(j)]['veclabel'],1.0/len(b.node[hb.getNodeIndex(j)]['veclabel']))#self.beta)
                            k+=self.computeSubtreeVecLabelKernels(a,ha.getNodeIndex(i),b,hb.getNodeIndex(j))*self.l**ha.getSubtreeSize(ci)
                        else:
                            k += self.l**ha.getSubtreeSize(ci)
                        j += 1
                    i += 1
            elif ha.getSubtreeID(i) < hb.getSubtreeID(j):
                i += 1
            else:
                j += 1
        return k

    def __str__(self):
        return "Subtree Kernel with lambda=" + self.l
        
        
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
        



## GRAPH FUNCTIONS
class SubtreeIDSubtreeSizeList():
    def __init__(self,G, root):
        self.sids = computeSubtreeIDSubtreeSizeList(G, root)

    def getSubtreeID(self,i):
        return self.sids[i][0]

    def getSubtreeSize(self,i):
        return self.sids[i][1]
    def getNodeIndex(self,i):
        return self.sids[i][2]

    def sort(self):
        self.sids.sort()

    def __len__(self):
        return len(self.sids)

def computeSubtreeIDSubtreeSizeList(T,root):
        #compute a list of pairs (subtree-hash-identifiers, subtree-size)

        p = [(setHashSubtreeIdentifier(T,root), getSubtreeSize(T,root),root)]
        for c in T.node[root]['childrenOrder']:
            p.extend(computeSubtreeIDSubtreeSizeList(T,c))
        return p

def getSubtreeSize(T,nodeID): 
        """The method returns the number of nodes in the subtree rooted at self"""
        if 'subtreeSize' in T.node[nodeID]:
            return T.node[nodeID]['subtreeSize']
        n = 1
        for c in T.node[nodeID]['childrenOrder']:
            n += getSubtreeSize(T,c)
        T.node[nodeID]['subtreeSize']=n    
        return n

def setHashSubtreeIdentifier(T, nodeID, sep='|'):
        """
        The method computes an identifier of the node based on
        1) the label of the node self
        2) the hash values of the children of self
        For each visited node the hash value is stored into the attribute subtreeId
        The label and the identifiers of the children nodes are separated by the char 'sep'
        """
        #print "Hash subtree calculation"
        #TODO tipo di ordine
        #assume ordered children
        if 'subtreeID' in T.node[nodeID]:
            return T.node[nodeID]['subtreeID']
        stri = str(T.node[nodeID]['label'])
        if stri.find(sep) != -1:
            print "ERROR: identifier " + sep + "used in label. Please set it with setHashSep(newsep)"
        for c in T.node[nodeID]['childrenOrder']:#T.successors(nodeID):
            #print "children exists"            
            stri += sep + setHashSubtreeIdentifier(T,c,sep)
        #print stri
        T.node[nodeID]['subtreeID'] = str(hash(stri)) #hash()
        return T.node[nodeID]['subtreeID']








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
        #print "setOrder Veclabels"
        #print "sep",sep

        if 'orderString' in T.node[nodeID]:
            return T.node[nodeID]['orderString']
        stri = str(T.node[nodeID]['label'])

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


#def setHashSubtreeIdentifier(T, nodeID, sep='|',labels=True):
#        """
#        The method computes an identifier of the node based on
#        1) the label of the node self
#        2) the hash values of the children of self
#        For each visited node the hash value is stored into the attribute subtreeId
#        The label and the identifiers of the children nodes are separated by the char 'sep'
#        """
#        #print "labels",labels
#
#        if 'subtreeID' in T.node[nodeID]:
#            return T.node[nodeID]['subtreeID']
#        if labels:
#            stri = str(T.node[nodeID]['label'])
#        else:
#            stri = str(T.out_degree(nodeID))
#        if stri.find(sep) != -1:
#            print "ERROR: identifier " + sep + "used in label. Please set it with setHashSep(newsep)"
#        for c in T.node[nodeID]['childrenOrder']:#T.successors(nodeID):
#            stri += sep + setHashSubtreeIdentifier(T,c,sep)
#        T.node[nodeID]['subtreeID'] = str(hash(stri))
#        return T.node[nodeID]['subtreeID']


#def computeSubtreeIDSubtreeSizeList(self):
#        #compute a list of pairs (subtree-hash-identifiers, subtree-size)
#        if not self:
#            return
#        p = [(self.subtreeId, self.stsize)]
#        for c in self.chs:
#            p.extend(c.computeSubtreeIDSubtreeSizeList())
#        return p
#
#
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
    def sort(self):
        self.productionlist.sort(cmp = lambda x, y: cmp(x[0], y[0]))
        self.productionlist.sort(cmp = lambda x, y: cmp(len(x[0]), len(y[0])))


    def __len__(self):
        return len(self.productionlist)

    def compareprods(x,y):
        if len(x[0])==len(y[0]): 
            return cmp(x[0],y[0])
        else:
            return cmp(len(x[0]),len(y[0]))


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