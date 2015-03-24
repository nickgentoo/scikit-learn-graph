
"""
A set of Classes to handle tree structured data
"""

import random
import bisect
import sys

class TreeNode():
    """
    A simple class for handling tree nodes
    """

    def __init__(self,val=None,chs=[]):
        """
        The constructor of TreeNode optionally receives two arguments:
        1) the label of the node
        2) the list of TreeNode objects representing the children of the node
        """
        self.val = str(val) #node label
        self.chs = chs      #list of children of the node

    def tostring_prolog(self):
        """
        Returns a string in which the subtree rooted
        at self is represented in prolog-style.  
        """
        if not self:
            return
        stri = ""
        if self.chs:
            stri += "%s("%self.val
            for i,c in enumerate(self.chs):
                stri += c.tostring_prolog()
                if i<len(self.chs)-1:
                    stri += ","
            stri += ")"
        else:
            stri += "%s"%self.val
        return stri

    def __str__(self): 
        """
        The method returns a string representing the subtree rooted at current node.
        Currently the string is in prolog format
        """
        return self.tostring_prolog()

    def tostring_svmlight(self): 
        """
        The method returns a string in which the subtree rooted
        at self is represented in svmlight-style 
        """
        if not self:
            return
        stri = ""
        if self.chs:
            stri += "(%s "%self.val
            for i,c in enumerate(self.chs):
                stri += c.tostring_svmlight()
            stri += ")"
        else:
            stri += "(%s -)"%self.val
        return stri

    def getNodeLabel(self):
        """Returns the label of the node"""
        return self.val

    def setNodeLabel(self, newlabel):
        self.val = newlabel

    def getChild(self,i):
        """
        Returns the TreeNode object related to the i-th child of the node. 
        If such node does not exists the value [] is returned and no error is issued. 
        """
        if not self or i>=len(self.chs):
            return []
        return self.chs[i]

    def getChildren(self):
        """The method returns the list of TreeNode objects representing the children of the node."""
        return self.chs

    def getOutdegree(self):
        """The method returns the number of children of the node"""
        if not self:
            return 0
        else:
            return len(self.chs)

    def getMaxOutdegree(self):
        """
        The method returns the maximum outdegree of the subtree rooted at node.
        If self is a leaf node, then 0 is returned and no error message is issued. 
        """
        if not self:
            return 0
        else:
            m = self.getOutdegree()
            for c in self.chs:
                m = max(m, c.getMaxOutdegree())
        return m

    def labelList(self): 
        """
        The method returns, for all descendants of the node, a triplet composed by 
        the label, a reference to the node, the index of the node w.r.t its siblings.
        """
        if not self: return []
        p = [(self.val,self, self.ind)]
        for c in self.chs:
            p.extend(c.labelList())
        return p

    def getProduction(self): 
        """
        The method returns a string representing the label of the current node (self) concatenated with the labels of its children
        The format of the string is the following: l_v(l_ch1,l_ch2,...,l_chn)
        where l_v is the label of self and l_chi is the label of the i-th child. 
        For example the string representing a subtree composed by a node labelled with A and two children labelled as B and C, 
        is represented as A(B,C)
        The empty string is returned in case the node is not a TreeNode object properly initialized. 
        """
        if not self: return ""
        self.production = self.val + "(" + ','.join([c.val for c in self.chs]) + ")"
        return self.production

    def productionlist(self): 
        """
        The method returns the list of productions of all nodes in the subtree rooted at node
        """
        if not self: return []
        p = [(self.getProduction(),self)]
        for c in self.chs:
            p.extend(c.productionlist())
        return p

    def getSubtreeSize(self): 
        """The method returns the number of nodes in the subtree rooted at self"""
        if not self:
            return 0
        n = 1
        for c in self.chs:
            n += c.getSubtreeSize()
        return n

    def setSubtreeSize(self):
        """
        The method returns the number of nodes in the subtree rooted at self
        for each visited node A such value is stored in A.stsize
        """
        if not self:
            self.stsize = 0
            return 0
        n = 1
        for c in self.chs:
            n += c.setSubtreeSize()
        self.stsize = n
        return n

    def getDepth(self):
        """
        The method returns the value of the attribute depth of the node. 
        The depth is supposed to have computed before calling this method (using setDepth).
        In case the depth has not been set, an error message is issued
        """
        if not hasattr(self, 'depth'):
            print "ERROR: node depth has not been computed!"
            return ""
        return self.depth

    def setDepth(self, subtreerootdepth = 1):
        """
        The method computes the depth (w.r.t self) of the descendants of self. The depth of self is 1 
        """
        if not self:
            return
        self.depth = subtreerootdepth
        for c in self.chs:
            c.setDepth(subtreerootdepth + 1)
        return
    
    def height(self): 
        """
        The method returns the length of the longest path connecting self to its farthest leaf
        """
        if not self:
            return 0
        p = 0
        for c in self.chs:
            p=max(p,c.height())
        return p+1

    def getLabelFrequencies(self):
        """
        The method computes a hash table 
        where the keys are the labels of the nodes in the subtree rooted at self 
        and the values are the frequencies of such labels.
        """
        lab = {}
        lab[self.val] = 1
        for c in self.chs:
            l = c.getLabelFrequencies()
            for lk in l.keys():
                if not lk in lab:
                    lab[lk] = l[lk]
                else:
                    lab[lk] += l[lk] 
        return lab

    def getSubtreeID(self):
        """
        The method returns the value of the attribute subtreeId, a string supposed to encode 
        a unique representation of the proper subtree rooted at node. Note that isomorphic 
        subtrees will get the same ID. 
        If the attribute subtreeId is not defined, no error message is issued.
        """
        return self.subtreeId

    def getParentNode(self):
        return self.parent

    def setParentNode(self):
        """
        The method adds a reference to the parent node for each node in the subtree rooted at self
        """
        if not self: return
        for c in self.chs:
            c.parent = self
            c.setParentNode()

    def setNodeIndex(self): 
        """The method assigns an index to the node w.r.t its siblings, i.e. the i-th child gets index i"""
        if not self: return
        i = 0
        for c in self.chs:
            i += 1
            c.ind = i
            c.setNodeIndex()

    def getHashSubtreeIdentifier(self, sep): 
        """
        The method computes and returns an identifier of the node based on
        1) the label of the node self
        2) the hash values of the children of self
        The label and the identifiers of the children nodes are separated by the char 'sep'
        """
        if not self:return
        stri = self.val
        for c in self.chs:
            stri += sep + c.getHashSubtreeIdentifier()
        return str(hash(stri))

    def setHashSubtreeIdentifier(self, sep):
        """
        The method computes an identifier of the node based on
        1) the label of the node self
        2) the hash values of the children of self
        For each visited node the hash value is stored into the attribute subtreeId
        The label and the identifiers of the children nodes are separated by the char 'sep'
        """
        if not self:return
        stri = self.val
        if stri.find(sep) != -1:
            print "ERROR: identifier " + sep + "used in label. Please set it with setHashSep(newsep)"
        for c in self.chs:
            stri += sep + c.setHashSubtreeIdentifier(sep)
        self.subtreeId = str(hash(stri))
        return self.subtreeId

    def computeSubtreeIDSubtreeSizeList(self):
        #compute a list of pairs (subtree-hash-identifiers, subtree-size)
        if not self:
            return
        p = [(self.subtreeId, self.stsize)]
        for c in self.chs:
            p.extend(c.computeSubtreeIDSubtreeSizeList())
        return p

    def computeSubtreePositionIDLabelSubtreeSizeList(self, h):
        #compute a hash whose key is the subtree-position-identifier and the value
        #is a triplet (subtree-hash-identifiers, node depth, subtree-size)
        #A key is constructed for each node
        if not self:
            return
        p = {}
        p[h] = (self.subtreeId, self.getDepth(), self.stsize)
        i = -1
        for c in self.chs:
            i += 1
            p.update(c.computeSubtreePositionIDLabelSubtreeSizeList(str(hash(h + "#" + str(i)))))
        return p


    def computeSubtreeIDTreeNodeList(self):
        if not self:
            return
        p = [(self.subtreeId, self)]
        for c in self.chs:
            p.extend(c.computeSubtreeIDTreeNode())
        return p

#######################################

class TreeNodeFromPrologString(TreeNode):
    """
    A class for creating TreeNode objects starting from strings in prolog format.
    The prolog format is as follows: 
    1) [a-zA-Z0-9]+: the label of the node
    2) (: beginning of a subtree (if there is any)
    3) ,: beginning of a sibling node (if there is any)
    4) ): end of a subtree (if there is any)
    Note that the chars (), can be changed by the user.
    """
    def __init__(self, s, symboltable="(),"):
        """
        This method creates a set of TreeNode objects corresponding to a subtree as described by the string s.
         : Parameter
        s: the string encoding the subtree
        symboltable (optional): a 3-char string where the first char corresponds to the beginning of a new subtree, 
                                the second corresponds to the end of a subtree and the third separates two siblings
        """
        startSubtreeChar,endSubtreeChar,startSiblingChar=symboltable
        s = s.rstrip('\n') #remove trailing newlines
        i, lens = (0, len(s))
        aa = []
        while (i < lens): 
            tmps = ""
            while (i < lens) and s[i] not in symboltable:
                tmps += s[i]
                i += 1
            if len(tmps) > 0:
                t = TreeNode(tmps,[])
                if len(aa)>0:
                    aa[-1].chs.append(t)
            if i < lens:
                if s[i] == startSubtreeChar: 
                    aa.append(t)
                elif s[i] == endSubtreeChar:
                    t=aa.pop()
                elif s[i] == startSiblingChar: 
                    pass
            i += 1
        self.val = t.getNodeLabel()
        self.chs = t.getChildren()

################################################

class TreeNodeFromSvmlightString(TreeNode):

    def __init__(self, s, symboltable="() "):
        """
        This method creates a set of TreeNode objects corresponding to a subtree as described by the string s.
         : Parameter
        s: the string encoding the subtree
        symboltable (optional): a 3-char string where the first char corresponds to the beginning of a new subtree, 
                                the second corresponds to the end of a subtree and the third separates the label 
                                of a node from its children
        """
        startSubtreeChar,endSubtreeChar,startSiblingChar=symboltable
        s = s.rstrip('\n') #remove trailing newlines
        i, lens = (0, len(s))
        aa = []
        while (i < lens): 
            tmps = ""
            while (i < lens) and s[i] not in symboltable:
                tmps += s[i]
                i += 1
            if len(tmps) > 0:
                t = TreeNode(tmps,[])
                if len(aa)>0:
                    aa[-1].chs.append(t)
            if i < lens:
                if s[i] == startSubtreeChar: 
                    aa.append(t)
                elif s[i] == endSubtreeChar:
                    t=aa.pop()
                elif s[i] == startSiblingChar: 
                    pass
            i += 1
        self.val = t.getNodeLabel()
        self.chs = t.getChildren()

################################################

class RandomTrees(TreeNode):
    """
    A class for generating random trees. The class depends on the following parameters:
    p, d: the two parameters are related to the probability of attaching a further child to the current node.
    outdegree: the maximum outdegree of any node
    nodelabels: the set of labels a node can get

    Each node has p*(d^x) probability of being created, where x is the depth of the node (the root node having depth x=0)
    If the node is created, then a label is randomly selected with uniform probablity between the set self.nodelabels and 
    outdegree children are attempted to be created with probability p*(d^(x+1)). The i-th successful attempt 
    creates the i-th child of the node (there cannot be a i-th non null child node while the j-th child, with j<i, is a null node). 
    """
    def __init__(self,p,d,outdegree,nodelabels):
        self.p = p
        self.d = d
        self.outdegree = outdegree
        self.nodelabels = nodelabels

    def __newTree(self, p):
        """internal method"""
        if random.random() > p:
            return None
        chs = []
        for i in range(self.outdegree):
            t = self.__newTree(p*self.d)
            if t: chs.append(t)
        return TreeNode(self.randomLabel(), chs)

    def newTree(self):
        """ 
        Create and returns one random subtree, i.e. a TreeNode object. Note that each returned tree has at least one node. 
        """
        t = self.__newTree(self.p)
        while not t: 
            t = self.__newTree(self.p)
        return t

    def randomLabel(self):
        """
        The method returns a label randomly selected, with uniform probability, between the candidate set of labels (self.nodelabels)
        """
        return random.choice(self.nodelabels)


class RandomTreesPowerLawDistribution(RandomTrees):
    """
    A class for generating random trees. It is a subclass of RandomTrees in which labels are selected
    randomly according zipf distribution (the first elements of node.labels have much higher probability 
    to be selected than the last ones). Here it is assumed that the labels are
    """
    def __init__(self, p, d, outdegree, numberoflabels):
        """
        The constructor takes 
        """
        RandomTrees.__init__(self,p,d,outdegree,[])
        s = 0.99
        self.nodelabels = [ 1/(i**s) for i in range(1,numberoflabels+1) ]
        norm = sum(self.nodelabels)
        self.nodelabels = [ x/norm for x in self.nodelabels ]
        cpd = 0
        for i in range(0,numberoflabels):
            cpd += self.nodelabels[i]
            self.nodelabels[i] = cpd

    def randomLabel(self):
        r = bisect.bisect(self.nodelabels,random.random())
        return r


####################################################################################################

class Tree():
    """
    A tree instance suitable for being processed by a tree kernel
    A TreeNode retain properties of a single node, a Tree a property
    of a set of nodes: target class, max depth, etc...
    Attributes:
    1) root: denotes the root node of the tree
    2) target (optional): target class for supervised learning
    """
    def __init__(self, root, target = 0):
        self.root = root
        self.target = target

    def deleteRootTreeNode(self):
        self.root = None

    def getTarget(self):
        """The method returns the value of the target attribute, which can be the empty string"""
        return self.target

    def getClass(self):
        """The method returns the value of the target attribute if not null, otherwise returns 0"""
        if not self.target:
            return 0
        return self.target

    def getMaxDepth(self):
        if not hasattr(root, 'maxdepth'):
            return self.root.height()
        else:
            return self.maxdepth

    def computeNodesDepth(self):
        self.root.setDepth()

    def setMaxDepth(self):
        self.maxdepth=self.root.height()

    def getMaxOutdegree(self):
        if not self.root:
            return 0 #ERROR?
        else:
            return self.root.getMaxOutdegree()

    def getLabelFrequencies(self):
        if not self.root:
            return {}
        else:
            return self.root.getLabelFrequencies()

    def __str__(self):
        if self.target:
            return str(int(self.target)) + " " + str(self.root) ###
        else:
            return str(self.root)            

    def printFormat(self, frmt = "prolog"):
        s = ""
        if self.target:
            s = str(self.target) + " "
        if frmt == "prolog":
            s += self.root.tostring_prolog()
        elif frmt == "svmlight":
            s += "|BT| " + self.root.tostring_svmlight() + " |ET| "
        return s

    def computeSubtreeIDs(self,hashsep):
        self.root.setHashSubtreeIdentifier(hashsep)

    def setParentNodes(self):
        """The method sets, for each node of the subtree rooted at self, a reference to its parent node"""
        self.root.parent = ""
        self.root.setParentNode()

    def setNodeIndices(self):
        self.root.ind = 0
        self.root.setNodeIndex()


#############################################

class TreeFromPrologString(Tree):
    """A Class for creating a Tree object from a string in prolog format."""
    def __init__(self, s):
        """
        Create a Tree object from a string representing a tree in prolog format (see below for a description of the format)
         : Parameter
        s: the string encoding the subtree and its label. The format of the string is the following
           1) [+-0-9]: numerical target label
           2) " "|\t 
           3) subtree in prolog format (see TreeNodeFromPrologString class for a description).
        """
        target, i, tmps = ("", 0, "")
        while s[i] in ".-+0123456789": #looking for numeric target value
            tmps += s[i]
            i += 1
        if len(tmps) > 0 and (s[i] == " " or s[i] == "\t"): #the target is valid
            target = tmps
            i+=1
        else:
            i=0
        self.root = TreeNodeFromPrologString(s[i:])
        self.target = float(target)

 
class SubtreeIDSubtreeSizeList():
    def __init__(self, root):
        self.sids = root.computeSubtreeIDSubtreeSizeList()

    def getSubtreeID(self,i):
        return self.sids[i][0]

    def getSubtreeSize(self,i):
        return self.sids[i][1]

    def sort(self):
        self.sids.sort()

    def __len__(self):
        return len(self.sids)

class ProdSubtreeList():
    def __init__(self, root):
        self.prodorderedlist = root.productionlist()

    def getProduction(self,i):
        return self.prodorderedlist[i][0]

    def getTree(self,i):
        return self.prodorderedlist[i][1]

    def sort(self):
        self.prodorderedlist.sort(cmp = lambda x, y: cmp(x[0], y[0]))
        self.prodorderedlist.sort(cmp = lambda x, y: cmp(len(x[0]), len(y[0])))

    def __len__(self):
        return len(self.prodorderedlist)

    def compareprods(x,y):
        if len(x[0])==len(y[0]): 
            return cmp(x[0],y[0])
        else:
            return cmp(len(x[0]),len(y[0]))

class LabelSubtreeList():
    """The class builds the data structure used by the route kernel"""
    def __init__(self, root):
        self.labelList = root.labelList()

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


class SubtreePositionIDLabelSubtreeSizeList():
    def __init__(self, root):
        self.sids = root.computeSubtreePositionIDLabelSubtreeSizeList(str(hash('0')))

    def getSubtreeID(self,i):
        return self.sids[i][0]

    def getNodeLabel(self,i):
        return self.sids[i][1]

    def getSubtreeSize(self,i):
        return self.sids[i][2]

    def __len__(self):
        return len(self.sids)

####################################################################################################

class Dataset():
    """
    A class for handling a collection of Objects.
    """
    def __init__(self, exampleList = []):
        self.examples = exampleList
        self.filename = ""

    def __str__(self):
        s = ""
        if self.filename:
            s += "Dataset loaded from file: " + self.filename + " "
        s += "Number of Examples: " + str(len(self)) + "\n"
        return s

    def loadFromFile(self, filename):
        """
        Load a set of strings from a file and creates a list of examples objects. File format is specified by the method loadExample().
         : Parameter
        filename: the name of the file containing the set of strings in prolog format
        """
        self.filename = filename
        self.examples = []
        f = open(filename,"r")
        for line in f:
            self.addExample(self.loadExample(line))
        f.close()

    def __len__(self):
        """The method returns the number of examples in the dataset"""
        return len(self.examples)

    def addExample(self, example):
        self.examples.append(example)

    def getExample(self, i):
        """The method returns the i-th member of the dataset. Note that the first example has index 0."""
        return self.examples[i]

    def getTargetList(self):
        """The method returns the list of target labels of all the examples. Note that the first example has index 0."""
        return [ e.getTarget() for e in self.examples ]
        
    def loadExample(self, line):
        pass

    def random_permutation(self, seed):
        pass

    def getStats(self):
        pass

    def printToFile(self, filename):
        """
        Given the name of a file as a parameter, the method prints the whole dataset to it (one example per line).
        The file is overwritten. 
        """
        f = open(filename,"w")
        for i in range(len(self.examples)):
            f.write(str(self.examples[i]) + "\n")
        f.close()

    def printToFileSvmlightFormat(self, filename):
        f = open(filename,"w")
        for i in range(len(self.examples)):
            f.write(str(self.examples[i].printFormat("svmlight")) + "\n")
        f.close()
        
    def merge(self, dataset, indices=""):
        """Merge the dataset with the one given as parameter."""
        if indices=="":
            indices = range(len(dataset))
        for i in indices:
            self.addExample(dataset.getExample(i))

######################################################################################################

class Datasets():
    """A class for handling collections of datasets"""
    def __init__(self, datasets=[]):
        self.datasets = datasets
        self.filenames = []
        for i in range(len(datasets)):
            self.filenames.append("")

    @classmethod
    def addFromFileNames(cls, filenames):
        """The method, given a list of file names, creates an array of dataset objects."""
        if len(filenames)==0:
            sys.error("Trying to create a Datasets object from an empty list of filenames")
        datasets = []
        for filename in filenames:
            datasets.append(self.datasetFromFileName(filename))
        return cls(Datasets.__init__(datasets))

    def addDatasetsFromFileNames(self, filenames):
        """The method loads multiple datasets from their filenames."""
        for f in filenames:
            self.addDataset(self.datasetFromFileName(f))

    def datasetFromFileName(self, filename):
        """
        The method loads a single dataset from a filename.
        """
        td = self.createDataset()
        td.loadFromFile(filename)
        return td

    def createDataset(self):
        """@abstract - The method creates a single dataset (which usually is a fusion of multiple datasets in Datasets."""
        sys.error("The method must be implemented in subclasses")

    def getDataset(self, i):
        """The method returns the i-th datasets. The first dataset has index 0."""
        if i >= len(self):
            sys.error("")
        return self.datasets[i]
        
    def __len__(self):
        """The method returns the number of datasets in the Datasets object."""
        return len(self.datasets)

    def totalNumberOfExamples(self):
        return sum([len(d) for d in self.datasets])
        
    def mergeDatasets(self, datasetindices, exampleindices=""):
        """The method, given a set of indices, merges the corresponding datasets and create a further dataset object."""
        dat = self.createDataset()
        if len(datasetindices)==len(exampleindices):
            for i in range(len(datasetindices)):
                dat.merge(self.getDataset(datasetindices[i]), exampleindices[i])
        else:
            for i in datasetindices:
                dat.merge(self.datasets[i], "")
        return dat

    def addDataset(self, dataset, filename=""):
        """
        The method adds a new dataset to the current set of datasets.
         : Parameter
        dataset: a Dataset object
        """
        self.datasets.append(dataset)
        self.filenames.append(filename)

#################################

class TreeDatasets(Datasets):
    """A class for handling collections of tree datasets"""
    def createDataset(self):
        return TreeDataset([])

class TreeDatasetsPrologFormat(Datasets):
    """A class for handling collections of tree datasets"""
    def createDataset(self):
        return TreeDatasetPrologFormat([])

######################################################################################################

class TreeDataset(Dataset):
    """A class for handling tree datasets."""

    def loadExample(self, line):
        return TreeFromPrologString(line)

    def generateRandomDataset(self, randObj, numberofexamples):
        """ 
        The method generates a random dataset
        """
        self.examples = []
        for i in range(numberofexamples):
            self.examples.append(Tree(randObj.newTree(),1))

    def getLabelFrequencies(self):
        """The method returns a dictionary where the keys are node labels and values their frequencies in all dataset."""
        lab = {}
        for i in range(len(self.examples)):
            l = self.examples[i].getLabelFrequencies()
            for lk in l.keys():
                if lk not in lab:
                    lab[lk] = l[lk]
                else:
                    lab[lk] += l[lk]
        return lab

    def getTotalNumberOfNodes(self):
        if hasattr(self, 'totalnodes'):
            return self.totalnodes
        else:
            s = 0
            for i in range(len(self.examples)):
                s += self.examples[i].root.getSubtreeSize()
            return s

    def setTotalNumberOfNodes(self):
        self.totalnodes = self.getTotalNumberOfNodes()

    def getNodesNumberAverage(self):
        return self.getTotalNumberOfNodes()/len(self.examples)

    def getNodesNumberVariance(self):
        avg = self.getNodesNumberAverage()
        s = 0
        for i in range(len(self.examples)):
            s += (avg-len(self.examples[i]))**2
        return s/(len(self.examples))

    def getAverageMaxOutdegree(self):
        o = 0
        for i in range(len(self.examples)):
            o += self.examples[i].getMaxOutdegree()
        return o

    def getMaxMaxOutdegree(self):
        o = 0
        for i in range(len(self.examples)):
            o = max(o,self.examples[i].getMaxOutdegree())
        return o

    def getMaxAndAverageMaxOutdegree(self):
        o,m = (0,0)
        for i in range(len(self.examples)):
            cm = self.examples[i].getMaxOutdegree()
            o += cm
            m = max(m,cm)
        return o,m

    def getMaxDepth(self):
        o = 0
        for i in range(len(self.examples)):
            o = max(o, self.examples[i].getMaxDepth())
        return o

    def getStats(self):
        self.setTotalNumberOfNodes()
        avgo,maxo = self.getMaxAndAverageMaxOutdegree()
        #maxdepth = self.getMaxDepth()
        s = "%f %d %d %f %d" % (self.getNodesNumberAverage(), self.getTotalNumberOfNodes(),maxo,avgo) #, maxdepth)
        return s

###############################################

class TreeDatasetPrologFormat(TreeDataset):
    """A class for handling tree datasets from a set of strings in prolog format."""
    def loadExample(self, line):
        return TreeFromPrologString(line)

