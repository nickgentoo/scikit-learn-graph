import tree
import sys
import math
from copy import deepcopy

class Kernel():
    """Common routines for kernel functions"""

    def kernel(self, a, b):
        """compute the tree kernel on the trees a and b"""
        if not isinstance(a, tree.Tree):
            print "ERROR: first parameter has to be a Tree Object"
            return ""
        if not isinstance(b, tree.Tree):
            print "ERROR: second parameter has to be a Tree Object"
            return ""
        self.preProcess(a)
        self.preProcess(b)
        return self.evaluate(a,b)

    def preProcess(self, a):
        """
        @abtract - The method creates any data structure useful for computing the kernel. 
        It is usually called only once per example. 
        This method must be defined in subclasses
        """
        sys.exit("ERROR: the method preProcess() must be invoked in subclasses")

    def evaluate(self, a, b):
        """
        @abtract - The method computes the kernel function between the examples a and b
        This method must be defined in subclasses
        """
        sys.exit("ERROR: evaluated() must be executed in subclasses")

    def printKernelMatrix(self, dataset):
        """
        The method computes the kernel matrix from a Dataset object.
        It creates the attributes self.km and self.target which contain 
        the kernel matrix and the target values of the examples
        The matrix is encoded as a list of dictionaries. 
        Each dictionary has a element whose key is 0 and whose value is the index of the example
        """
        if not isinstance(dataset, tree.Dataset):
            print "ERROR: the first parameter must be a Dataset object"
            return
        ne = len(dataset)
        self.km=[]
        for i in range(ne):
            rowexample = dataset.getExample(i)
            krow = {}
            krow[0]=i+1
            for j in range(0,ne):
                krow[j+1]=self.kernel(rowexample, dataset.getExample(j))
            self.km.append(krow)
        self.target = dataset.getTargetList()

    def __str__(self):
        """@abtract - The method describes the kernel instance and its parameters."""
        sys.exit("ERROR: the method __str__ must be invoked in subclasses")

    def setParameters(self, *params):
        """
        The methods sets kernel parameters. 
         : Parameter
        params: a dictionary whose keys are parameter names and values their parameter values
        """
        sys.exit("ERROR: the method setParameters must be invoked in subclasses")

##########################################################################################################

class KernelMatrix():
    """The base class for handling kernel matrices"""

    def __init__(self, km):
        self.km = km

    @classmethod
    def compute(cls, dataset, kernelfunction):
        """
        The method computes the kernel matrix from a Dataset object.
        It creates the attributes self.km and self.target which contain 
        the kernel matrix and the target values of the examples
        The matrix is encoded as a list of dictionaries. 
        """
        if not isinstance(dataset, tree.Dataset):
            print "ERROR: the first parameter must be a Dataset object"
            return ""
        if not isinstance(kernelfunction, Kernel):
            print "ERROR: second parameter must be a Kernel object"
            return ""
        ne = len(dataset)
        km=[]
        for i in range(ne):
            rowexample = dataset.getExample(i)
            krow = {}
            for j in range(0,ne):
                krow[j] = kernelfunction.kernel(rowexample, dataset.getExample(j))
            km.append(krow)
        t=cls(km)
        t.dataset, t.kernelfunction = (dataset, kernelfunction)
        return t

    def getKernelMatrix(self):
        return self.km

    def getKernelMatrixRow(self, rownumber):
        return self.km[rownumber]

    def __len__(self):
        return len(self.km)

    def Print(self, filename=""):
        """
        The method prints the kernel matrix to file or standard output if no parameter is given. 
        Each line of output corresponds to the evaluation of the kernel function for examples i and j.
        More precisely, the output is a triplet, separated by spaces:i j k(i,j) 
        Only the upper triangular part of the kernel matrix is printed
        """
        if filename=="":
            f = sys.stdout
            closefile = 0
        else:
            f = open(filename, "w")
            closefile = 1
        ne = len(self.km)
        for i in range(ne):
            for j in range(i,ne):
                f.write("%d %d %s\n" % (i+1, j+1, str(self.km[i][j])))
        if close==1: 
            f.close()
            print "Kernel Matrix printed to file: " + filename

    #def __str__(self):
    #    return "Kernel Matrix of dataset %s, obtained from kernel %s"%(self.)
##########################################################################################################

class KernelMatrixLibsvm(KernelMatrix):
    """
    A subclass of KernelMatrix especially suited to interact with Libsvm software. 
    """
    def __init__(self, km, target):
        KernelMatrix.__init__(self, km)
        for i in range(0, len(self.km)):
            self.km[i][0] = i+1
        self.target = target

    @classmethod
    def compute(cls, dataset, kernelfunction):
        """
        The method computes the kernel matrix from a Dataset object.
        It creates the attributes self.km and self.target which contain 
        the kernel matrix and the target values of the examples
        The matrix is encoded as a list of dictionaries. 
        Each dictionary has a element whose key is 0 and whose value is the index of the example
        """
        if not isinstance(dataset, tree.Dataset):
            print "ERROR: the first parameter must be a Dataset object"
            return ""
        if not isinstance(kernelfunction, Kernel):
            print "ERROR: second parameter must be a Kernel object"
            return ""
        ne = len(dataset)
        km=[]
        for i in range(1, ne+1):
            rowexample = dataset.getExample(i-1)
            krow = {}
            krow[0] = i
            for j in range(1, ne+1):
                krow[j] = kernelfunction.kernel(rowexample, dataset.getExample(j-1))
            km.append(krow)
        t=cls(km, dataset.getTargetList())
        t.dataset, t.kernelfunction = (dataset, kernelfunction)
        return t

    @classmethod
    def loadFromLIBSVMFile(cls, filename):
        """
        The method loads a kernel matrix in libsvm format from a file.
        It creates the attributes self.km and self.target which contain 
        the kernel matrix and the target values of the examples
        The matrix is encoded as a list of dictionaries. 
        Currently, the whole kernel matrix is encoded.
        -Typical usage: a = tree_kernels.KernelMatrixLibsvm.loadFromLIBSVMFile("myfile")
        """
        nexamples = sum(1 for line in open(filename,"r"))
        target = [int(0)] * nexamples
        km = [0] * nexamples
        f = open(filename,"r")
        for line in f:
            i,lenline = (0, len(line))
            tmps=""
            while i< lenline and line[i] in "-+0123456789": #looking for numeric target value        
                tmps += line[i]
                i += 1
            if len(tmps) > 0 and (line[i] == " " or line[i] == "\t"): #the target is valid 
                targetvalue = int(tmps)
                i+=1
            else:
                print "Error in file format! No numerical target found"
                return ""
            if line[i]=="0" and line[i+1]==":":
                i += 2
                tmpind = ""
                while i<lenline and line[i] in "0123456789": 
                    tmpind+=line[i]
                    i+=1
                ind=int(tmpind)-1
            else:
                print "ERROR! index of example not found!"
                return ""
            krow={}
            krow[0]=ind 
            for x in line[i+1:].split(" "):
                y=x.split(":")
                if len(y)>1:
                    krow[int(y[0])] = float(y[1])
            km[ind] = krow
            target[ind] = targetvalue
        t = cls(km, target)
        f.close()
        return t

    def getMatrix(self):
        """The method returns the kernel matrix."""
        return self.km
    
    def getTargets(self, indices=""):
        """
        The method returns a list of target labels. 
         : Parameter 
        indices: the list of indices of the examples whose target label is returned; 
                 if indices=="" then the target labels of all the examples are returned. 
        """
        if indices=="":
            return self.target
        else:
            return [self.getTarget(i) for i in indices]

    def getTarget(self, i):
        """The method returns the target label of the i-th example."""
        return self.target[i]

    def Print(self, filename=""):
        """
        The method prints the kernel matrix in LIBSVM format to file or to standard output if no filename is given as parameter.
        """
        if filename=="":
            f = sys.stdout
            closefile = 0
        else:
            f = open(filename, "w")
            closefile = 1
        ne = len(self.km)
        for i in range(ne):
            f.write("%s" % (self.target[i]))
            for j in sorted(self.km[i].keys()):
                if j==0:
                    f.write(" %s:%s" % (j, str(self.km[i][j])))
                else:
                    f.write(" %s:%f" % (j, self.km[i][j]))
            f.write("\n")
        if closefile==1: 
            f.close()
            print "Kernel Matrix printed to file: " + filename

    def compare(self, kmObj, tolerance=0.000003):
        """Tells whether two kernel matrices are equal (up to the given tolerance)"""
        ne = len(self)
        if not ne == len(kmObj):
            print "The two kernel matrices have different number of rows: %d and %d"%(ne, len(kmObj))
            return False
        ncols = self.getKernelMatrixRow(0)
        for i in range(ne):
            krow = self.getKernelMatrixRow(i)
            ktworow = kmObj.getKernelMatrixRow(i)
            if not ncols==len(krow) or not ncols==len(ktworow):
                print "Inconsistent number of columns on row %d, expected %d, found %d and %d"%(i+1,ncols,len(krow),len(ktworow))
                return False
            for col in krow.keys():
                if col not in ktworow.keys():
                    print "cannot find index %d in the second kernel matrix"%(col)
                    return False
                if abs(krow[col]-ktworow[col]) > tolerance:
                    print "The kernel value between examples %d,%d differ: %f and %f"%(i+1,col+1, krow[col], ktworow[col])
                    return False
        print "The two kernel matrices are identical!"
        return True

##############################################################################################

class KernelST(Kernel):
    def __init__(self,l,savememory=1,hashsep="#"):
        self.l = float(l)
        self.hashsep = hashsep
        self.savememory = savememory

    def preProcess(self,a):
        if hasattr(a,'kernelstrepr'): #already preprocessed
            return 
        if not hasattr(a.root, 'stsize'):
            a.root.setSubtreeSize()
        a.root.setHashSubtreeIdentifier(self.hashsep)
        a.kernelstrepr = tree.SubtreeIDSubtreeSizeList(a.root)
        a.kernelstrepr.sort()
        if self.savememory==1:
            a.deleteRootNode()
        
    def evaluate(self,a,b):
        #Assumes ha and hb are ordered list of pairs (subtreeid, subtreesize) 
        #a.kernelreprst,b.kernelreprst are checked or created in preProcess()
        ha, hb = (a.kernelstrepr, b.kernelstrepr)
        i,j,k,toti,totj = (0,0,0,len(ha), len(hb))
        while i < toti and j < totj:
            if ha.getSubtreeID(i) == hb.getSubtreeID(j):
                ci,cj=(i,j)
                while i < toti and ha.getSubtreeID(i)==ha.getSubtreeID(ci):
                    i += 1
                while j < totj and hb.getSubtreeID(j)==hb.getSubtreeID(cj):
                    j += 1
                k += (i-ci)*(j-cj)*(self.l**ha.getSubtreeSize(ci))
            elif ha.getSubtreeID(i) < hb.getSubtreeID(j):
                i += 1
            else:
                j += 1
        return k

    def __str__(self):
        return "Subtree Kernel with lambda=" + self.l

############################################################################################

class KernelSST(Kernel):

    def __init__(self,l,hashsep="#"):
        self.l = float(l)
        self.hashsep = hashsep
        self.cache = Cache()
    
    def preProcess(self,a):
        if hasattr(a,'kernelsstrepr'): #already preprocessed
            return 
        a.root.setHashSubtreeIdentifier(self.hashsep)
        a.kernelsstrepr = tree.ProdSubtreeList(a.root)
        a.kernelsstrepr.sort()

    def CSST(self,c,d):
        if c.getSubtreeID() < d.getSubtreeID():
            tmpkey = str(c.getSubtreeID()) + "#" + str(d.getSubtreeID())
        else:
            tmpkey = str(d.getSubtreeID()) + "#" + str(c.getSubtreeID()) 
        if self.cache.exists(tmpkey):
            return float(self.cache.read(tmpkey))
        else:
            prod = self.l
            nc = c.getOutdegree()
            if nc==d.getOutdegree():
                for ci in range(nc):
                    if c.getChild(ci).getProduction() == d.getChild(ci).getProduction():
                        prod *= (1 + self.CSST(c.getChild(ci),d.getChild(ci)))
                    else:
                        cid, did = (c.getChild(ci).getSubtreeID(),d.getChild(ci).getSubtreeID())
                        if cid < did:
                            self.cache.insert(str(cid) + str(did), 0)
                        else:
                            self.cache.insert(str(did) + str(cid), 0)
            self.cache.insert(tmpkey, prod)
        return float(prod)

    def evaluate(self,a,b):
        pa,pb=(a.kernelsstrepr, b.kernelsstrepr)
        self.cache.removeAll()
        i,j,k,toti,totj = (0,0,0,len(pa),len(pb))
        while i < toti and j < totj:
            if pa.getProduction(i) == pb.getProduction(j):
                ci,cj=(i,j)
                while i < toti and pa.getProduction(i)==pa.getProduction(ci):
                    j = cj
                    while j < totj and pb.getProduction(j)==pb.getProduction(cj):
                        k += self.CSST(pa.getTree(i),pb.getTree(j))
                        j += 1
                    i += 1
            elif len(pa.getProduction(i))<len(pb.getProduction(j)) or (len(pa.getProduction(i))==len(pb.getProduction(j)) and pa.getProduction(i) < pb.getProduction(j)):
                i += 1
            else:
                j += 1
        return k

    def __str__(self):
        return "Subset Tree Kernel, with lambda=" + self.l

#######################################################################################

class KernelPT(Kernel):
    def __init__(self,l,m,normalize=True,hashsep="#"):
        self.l = float(l)
        self.m = float(m)
        self.hashsep = hashsep
        self.normalize = normalize
        self.cache = Cache()

    def preProcess(self,a):
        if hasattr(a,'kernelptrepr'): #already preprocessed
            return 
        a.root.setHashSubtreeIdentifier(self.hashsep)
        a.setNodeIndices()
        a.kernelptrepr = tree.LabelSubtreeList(a.root)
        a.kernelptrepr.sort()
        if self.normalize:
            a.norm = 1.0
            b = deepcopy(a)
            a.norm = math.sqrt(self.evaluate(a,b))

    def DeltaSk(self, a, b,nca, ncb):
        DPS = [[0 for i in range(ncb+1)] for j in range(nca+1)]
        DP = [[0 for i in range(ncb+1)] for j in range(nca+1)]
        kmat = [0]*(nca+1)
        for i in range(1,nca+1):
            for j in range(1,ncb+1):
                if a.getChild(i-1).getNodeLabel() == b.getChild(j-1).getNodeLabel():
                    DPS[i][j] = self.CPT(a.getChild(i-1),b.getChild(j-1))
                    kmat[0] += DPS[i][j]
                else:
                    DPS[i][j] = 0
        for s in range(1,min(nca,ncb)):
            for i in range(nca+1): 
                DP[i][s-1] = 0
            for j in range(ncb+1): 
                DP[s-1][j] = 0
            for i in range(s,nca+1):
                for j in range(s,ncb+1):
                    DP[i][j] = DPS[i][j] + self.l*DP[i-1][j] + self.l*DP[i][j-1] - self.l**2*DP[i-1][j-1]
                    if a.getChild(i-1).getNodeLabel() == b.getChild(j-1).getNodeLabel():
                        DPS[i][j] = self.CPT(a.getChild(i-1),b.getChild(j-1))*DP[i-1][j-1]
                        kmat[s] += DPS[i][j]
        return sum(kmat)
    
    def CPT(self,c,d):
        if c.getSubtreeID() < d.getSubtreeID():
            tmpkey = str(c.getSubtreeID()) + self.hashsep + str(d.getSubtreeID())
        else:
            tmpkey = str(d.getSubtreeID()) + self.hashsep + str(c.getSubtreeID()) 
        if self.cache.exists(tmpkey):
            return self.cache.read(tmpkey)
        else:
            if c.getOutdegree()==0 or d.getOutdegree()==0:
                prod = self.m*self.l**2
            else:
                prod = self.m*(self.l**2+self.DeltaSk(c, d,c.getOutdegree(),d.getOutdegree()))
            self.cache.insert(tmpkey, prod)
            return prod     


    def evaluate(self,a,b):
        self.cache.removeAll()
        la,lb = (a.kernelptrepr,b.kernelptrepr)
        i,j,k,toti,totj = (0,0,0,len(la),len(lb))
        while i < toti and j < totj:
            if la.getNodeLabel(i) == lb.getNodeLabel(j):
                ci,cj=(i,j)
                while i < toti and la.getNodeLabel(i) == la.getNodeLabel(ci):
                    j = cj
                    while j < totj and lb.getNodeLabel(j) == lb.getNodeLabel(cj):
                        k += self.CPT(la.getTree(i),lb.getTree(j))
                        j += 1
                    i += 1
            elif la.getNodeLabel(i) <= lb.getNodeLabel(j):
                i += 1
            else:
                j += 1

        if self.normalize:
            k = k/(a.norm*b.norm)
        return k

    def __str__(self):
        return "Partial Tree Kernel, with lambda=%f and mu=%f" % (self.l, self.m)

################################################################################################

class KernelPdak(Kernel):
    def __init__(self, l, gamma, beta, hashsep="#"):
        self.l = float(l)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.hashsep = hashsep

    def preProcess(self, t):
        if hasattr(t,'kernelpdakrepr'): #already preprocessed
            return 
        if not hasattr(t.root, 'stsize'):
            t.root.setSubtreeSize()
        t.root.setHashSubtreeIdentifier(self.hashsep)
        t.computeNodesDepth()
        t.kernelpdakrepr = tree.SubtreePositionIDLabelSubtreeSizeList(t.root)

    def mergetrees_with_depth(self, tree1, tree2):
        merge = {}
        for key in tree1:
            if key in tree2:
                merge[key] = ({(tree1[key][0],tree1[key][2]):{tree1[key][1]:1}},{(tree2[key][0],tree2[key][2]):{tree2[key][1]:1}})
                del tree2[key]
            else: merge[key] = ({(tree1[key][0],tree1[key][2]):{tree1[key][1]:1}},None)
        for key in tree2:
            merge[key] = (None,{(tree2[key][0],tree2[key][2]):{tree2[key][1]:1}})
        return merge

    def visit_with_depth(self,jtree,node,depth,param,lambda_par,gamma_par):
        kvalue = 0
        if node is not None :
            child = 0
            key = str(hash(node+'#'+str(child)))

            while key in jtree :
                kvalue = kvalue + self.visit_with_depth(jtree,key,depth+1,param,lambda_par,gamma_par)
                if jtree[key][0] is not None:
                    if jtree[node][0] is None:
                        #jtree[node][0] = jtree[key][0]
                        jtree[node] = (jtree[key][0], jtree[node][1]) 
                    else:
                        for tmpkey in jtree[key][0]:
                            if tmpkey in jtree[node][0]:
                                for tmpkey2 in jtree[key][0][tmpkey]:
                                    if tmpkey2 in jtree[node][0][tmpkey]:
                                        jtree[node][0][tmpkey][tmpkey2] = jtree[node][0][tmpkey][tmpkey2] + jtree[key][0][tmpkey][tmpkey2]
                                    else: jtree[node][0][tmpkey][tmpkey2] = jtree[key][0][tmpkey][tmpkey2]    
                            else: jtree[node][0][tmpkey] = jtree[key][0][tmpkey]
                if jtree[key][1] is not None:
                    if jtree[node][1] is None:
                        #jtree[node][1]=jtree[key][1]
                        jtree[node]=(jtree[node][0],jtree[key][1]) 
                    else:
                        for tmpkey in jtree[key][1]:
                            if tmpkey in jtree[node][1]:
                                for tmpkey2 in jtree[key][1][tmpkey]:
                                    if tmpkey2 in jtree[node][1][tmpkey]:
                                        jtree[node][1][tmpkey][tmpkey2] = jtree[node][1][tmpkey][tmpkey2] + jtree[key][1][tmpkey][tmpkey2]
                                    else: jtree[node][1][tmpkey][tmpkey2] = jtree[key][1][tmpkey][tmpkey2]
                            else: jtree[node][1][tmpkey] = jtree[key][1][tmpkey]
                child = child + 1
                key = str(hash(node+'#'+str(child)))
            # print jtree[node]
            if (jtree[node][0] is not None) and (jtree[node][1] is not None):
                for lkey in jtree[node][0]:
                    if lkey in jtree[node][1]:
                        tmpk = 0
                        for fkey1 in jtree[node][0][lkey]:
                            for fkey2 in jtree[node][1][lkey]:
                                tmpk = tmpk + lambda_par**lkey[1]*jtree[node][0][lkey][fkey1]*jtree[node][1][lkey][fkey2]*math.exp(-param*(fkey1 + fkey2))
                        kvalue = kvalue + (gamma_par**depth)*tmpk*math.exp(2*param*depth)
            return kvalue


    def evaluate(self,a,b):
        tree1 = deepcopy(a.kernelpdakrepr.sids)
        tree2 = deepcopy(b.kernelpdakrepr.sids)
        m = self.mergetrees_with_depth(tree1,tree2)
        kvalue = self.visit_with_depth(m,str(hash('0')),1,self.l, self.gamma, self.beta)
        del m, tree1, tree2
        return kvalue

    def __str__(self):
        return "Position Aware Tree Kernel, with lambda=" + self.l + " gamma=" + self.gamma + " and beta=" + self.beta

################################################################################################

class KernelRoute(Kernel):
    """
    The class computes the Route Kernel as described in the paper:
    F. Aiolli, G. Da San Martino, and A. Sperduti, "Route kernels for trees" 
    in Proceedings of the 26th Annual International Conference on Machine Learning. Montreal, Quebec, Canada: ACM, 2009, pp. 17-24.
    """
    def __init__(self,l):
        self.l = float(l)

    def preProcess(self,a):
        if hasattr(a,'kernelrouterepr'): #already preprocessed
            return 
        if not hasattr(a.root, 'parent'):
            a.setParentNodes()
            a.setNodeIndices()
        a.kernelrouterepr = tree.LabelSubtreeList(a.root)

    def evaluate(self,a,b):
        """The method evaluates the base kernel"""
        ha, hb = (a.kernelrouterepr, b.kernelrouterepr)
        i,j,k,toti,totj = (0,0,0,len(ha), len(hb))
        while i < toti and j < totj:
            if ha.getNodeLabel(i) == hb.getNodeLabel(j):
                ci,cj=(i,j)
                while i < toti and ha.getNodeLabel(i)==ha.getNodeLabel(ci):
                    i += 1
                while j < totj and hb.getNodeLabel(j)==hb.getNodeLabel(cj):
                    j += 1
                k += (i-ci)*(j-cj)*(self.l) + self.evaluateKernelOnRoutes([ha.getTree(x) for x in range(ci, i) if ha.getIndex(x)>0], 
                                                                          [hb.getTree(x) for x in range(cj, j) if hb.getIndex(x)>0], 
                                                                          self.l*self.l)
            elif ha.getNodeLabel(i) < hb.getNodeLabel(j):
                i += 1
            else:
                j += 1
        return k

    def evaluateKernelOnRoutes(self, ta, tb, l):
        k, i, j, toti, totj= (0, 0, 0, len(ta), len(tb))
        ta.sort(cmp = lambda x, y: cmp(x.ind, y.ind))
        tb.sort(cmp = lambda x, y: cmp(x.ind, y.ind))
        while i < toti and j < totj:
            if ta[i].ind == tb[j].ind:
                ci,cj=(i,j)
                while i < toti and ta[i].ind==ta[ci].ind:
                    i += 1
                while j < totj and tb[j].ind==tb[cj].ind:
                    j += 1
                k += (i-ci)*(j-cj)*l + self.evaluateKernelOnRoutes([ ta[x].parent for x in range(ci, i) if ta[x].ind>0 ], 
                                                                   [ tb[x].parent for x in range(cj, j) if tb[x].ind>0 ],
                                                                   l*l)
            elif ta[i].ind < tb[j].ind:
                i += 1
            else:
                j += 1
        return k

    def __str__(self):
        return "Route Tree Kernel, with lambda=" + self.l

################################################################################################

class KernelRouteSST(Kernel):
    """The class computes the route kernel combined with the SST kernel."""
    def __init__(self,lroute,lsst,hashsep="#"):
        self.l = float(lroute)
        self.lsst = float(lsst)
        self.hashsep = hashsep
        self.cache = Cache()
    
    def preProcess(self,a):
        if hasattr(a,'kernelrouterepr'): #already preprocessed
            return 
        if not hasattr(a.root, 'parent'):
            a.setParentNodes()
            a.setNodeIndices()
        a.root.setHashSubtreeIdentifier(self.hashsep)
        a.kernelsstrepr = tree.ProdSubtreeList(a.root)
        a.kernelsstrepr.sort()
        a.kernelrouterepr = tree.LabelSubtreeList(a.root)

    def evaluate(self,a,b):
        """The method evaluates the base kernel"""
        ha, hb = (a.kernelrouterepr, b.kernelrouterepr)
        i,j,k,toti,totj = (0, 0, 0, len(ha), len(hb))
        while i < toti and j < totj:
            if ha.getNodeLabel(i) == hb.getNodeLabel(j):
                ci,cj=(i,j)
                while i < toti and ha.getNodeLabel(i)==ha.getNodeLabel(ci):
                    j = cj
                    while j < totj and hb.getNodeLabel(j)==hb.getNodeLabel(cj):
                        k += self.l*self.CSST(a.kernelsstrepr.getTree(i),b.kernelsstrepr.getTree(j))*\
                             self.evaluateKernelOnRoutes(ha.getTree(i), hb.getTree(j), self.l*self.l)
                        j += 1
                    i += 1
            elif ha.getNodeLabel(i) < hb.getNodeLabel(j):
                i += 1
            else:
                j += 1
        return k

    def evaluateKernelOnRoutes(self, ta, tb, l):
        if ta.ind ==0 and tb.ind == 0:
            return l
        if ta.ind == tb.ind:
            return l + self.evaluateKernelOnRoutes(ta.parent, tb.parent, l*l)
        return 0

    def CSST(self,c,d):
        if c.getSubtreeID() < d.getSubtreeID():
            tmpkey = str(c.getSubtreeID()) + "#" + str(d.getSubtreeID())
        else:
            tmpkey = str(d.getSubtreeID()) + "#" + str(c.getSubtreeID()) 
        if self.cache.exists(tmpkey):
            return float(self.cache.read(tmpkey))
        else:
            prod = self.lsst
            nc = c.getOutdegree()
            if nc==d.getOutdegree():
                for ci in range(nc):
                    if c.getChild(ci).getProduction() == d.getChild(ci).getProduction():
                        prod *= (1 + self.CSST(c.getChild(ci),d.getChild(ci)))
                    else:
                        cid, did = (c.getChild(ci).getSubtreeID(),d.getChild(ci).getSubtreeID())
                        if cid < did:
                            self.cache.insert(str(cid) + str(did), 0)
                        else:
                            self.cache.insert(str(did) + str(cid), 0)
            self.cache.insert(tmpkey, prod)
        return float(prod)

    def __str__(self):
        return "Route Kernel (lambda=" + self.l + ") where the base kernel is the Subset Tree Kernel (lambda=" + self.lsst + ")"


###############################

class newKernel(Kernel):
    """Template class for defining a novel tree kernel."""

    def preProcess(self, a):
        """
        @abtract - The method creates any data structure useful for computing the kernel. 
        It is usually called only once per example. 
        This method must be defined in subclasses
        """

    def evaluate(self, a, b):
        """
        @abtract - The method computes the kernel function between the examples a and b
        This method must be defined in subclasses
        """

    def __str__(self):
        """@abtract - The method describes the kernel instance and its parameters."""
        sys.exit("ERROR: the method __str__ must be invoked in subclasses")

    def setParameters(self, *params):
        """
        The methods sets kernel parameters. 
         : Parameter
        params: a dictionary whose keys are parameter names and values their parameter values
        """
        sys.exit("ERROR: the method setParameters must be invoked in subclasses")


################################################################################################

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


