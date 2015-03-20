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
import networkx as nx
if __name__=='__main__':
    if len(sys.argv)<2:
        sys.exit("python test_sst_inex.py  lambda_tree filename")
    la=float(sys.argv[1])
    name=str(sys.argv[2])

    #g_it=datasets.load_graphs_bursi()
    from skgraph.tree import tree_new
    from skgraph.dependencies.pythontk import tree_kernels_new
    g_it=[]
    from networkx import DiGraph
    graph1=nx.DiGraph()
    graph1.add_node(1,label="a")
    graph1.add_node(2,label="b")
    graph1.add_node(3,label="b")
    graph1.add_node(8,label="b")

    graph1.add_node(4,label="d")
    graph1.add_node(5,label="e")
    graph1.add_node(6,label="f")
    graph1.add_node(7,label="g")
    graph1.add_edge(1,8)

    graph1.add_edge(1,2)
    graph1.add_edge(1,3)
    graph1.add_edge(2,4)
    graph1.add_edge(2,5)
    graph1.add_edge(3,6)
    graph1.add_edge(3,7)
    graph1.graph['root']=1
    g_it.append(graph1)

    graph2=nx.DiGraph()
    graph2.add_node(1,label="a")
    graph2.add_node(2,label="b")
    graph2.add_node(3,label="b")
    graph2.add_node(8,label="b")

    graph2.add_node(4,label="d")
    graph2.add_node(5,label="e")
    graph2.add_node(6,label="o")
    graph2.add_node(7,label="p")
    graph2.add_edge(1,8)
    graph2.add_edge(1,2)
    graph2.add_edge(1,3)
    graph2.add_edge(2,4)
    graph2.add_edge(2,5)
    graph2.add_edge(3,6)
    graph2.add_edge(3,7)
    graph2.graph['root']=1

    g_it.append(graph2)
    
 
    treeKernel=tree_kernels_new.SSTKernel(la,veclabels=False)
    GM=treeKernel.computeKernelMatrix(g_it)
    np.savetxt(name,GM)
    treeKernelAM=tree_kernels_new.SSTAllMatchingKernel(la,veclabels=False)
    GM1=treeKernelAM.computeKernelMatrix(g_it)
    np.savetxt(name+"AllMatching",GM1)
    #print GM