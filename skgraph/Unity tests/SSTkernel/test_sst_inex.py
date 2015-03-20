__author__ = "Nicolo' Navarin"

import sys
import numpy as np
import time
from skgraph.ioskgraph import load_target
from skgraph import graph
from sklearn import svm
from sklearn import cross_validation
from skgraph.ODDCLGraphKernel import ODDCLGraphKernel
from skgraph import datasets

if __name__=='__main__':
    if len(sys.argv)<2:
        sys.exit("python test_sst_inex.py  lambda_tree  njobs filename")
    la=float(sys.argv[1])
    njobs=int(int(sys.argv[2]))
    name=str(sys.argv[3])

    #g_it=datasets.load_graphs_bursi()
    import tree_new
    from skgraph.dependencies.pythontk import tree_kernels_new
    g_it=tree_new.dataset_tree_inex2005()
    treeKernel=tree_kernels_new.SSTKernel(la)
    GM=treeKernel.computeKernelMatrix(g_it.trees)
    np.savetxt(name,GM)
    #print GM