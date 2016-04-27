# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:02:41 2015

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
along with scikit-learn-graph.  If not, see <http://www.gnu.org/licenses/>.
"""

import sys
# import time
from skgraph.kernel.WLGraphKernel import WLGraphKernel
from skgraph.kernel.ODDSTGraphKernel import ODDSTGraphKernel
from skgraph.kernel.NSPDKGraphKernel import NSPDKGraphKernel
from skgraph.kernel.ODDSTPGraphKernel import ODDSTPGraphKernel
from skgraph.kernel.WLCGraphKernel import WLCGraphKernel
from skgraph.kernel.ODDSTCGraphKernel import ODDSTCGraphKernel
from skgraph.kernel.ODDSTPCGraphKernel import ODDSTPCGraphKernel
from skgraph.kernel.WLDDKGraphKernel import WLDDKGraphKernel
from skgraph.kernel.WLNSKGraphKernel import WLNSKGraphKernel

from skgraph.datasets import load_graph_datasets
import numpy as np

if __name__=='__main__':
    if len(sys.argv)<2:
        sys.exit("python -m calculate_matrix_allkernels dataset r l filename kernel [normalization] [version] [norm_split] [normalization_type] [h]")

    # mandatory & fixed parameters

    dataset=sys.argv[1]
    max_radius=int(sys.argv[2])
    la=float(sys.argv[3])
    name=str(sys.argv[4])
    kernel=sys.argv[5]
    njobs=1

    # optional parameters

    # normalization as an integer encoded boolean [0|1]
    normalization = True
    if len(sys.argv) > 6:
        normalization = bool(sys.argv[6])

    # kernel version as an integer encoded enum [0|1] (default: 1)
    version = 1
    if len(sys.argv) > 7:
        version = int(sys.argv[7])

    # normalization split as an integer encoded bool [0|1]
    nsplit = False
    if len(sys.argv) > 8:
        nsplit = bool(sys.argv[8])

    # normalization type as an integer encoded enum [0|1|...]
    ntype = 0
    if len(sys.argv) > 9:
        ntype = int(sys.argv[9])

    # iteration count for WL extended kernels only, integer
    iterations = 1
    if len(sys.argv) > 10:
        iterations = int(sys.argv[10])
    
    if dataset=="CAS":
        print "Loading bursi(CAS) dataset"        
        g_it=load_graph_datasets.load_graphs_bursi()
    elif dataset=="GDD":
        print "Loading GDD dataset"        
        g_it=load_graph_datasets.load_graphs_GDD()
    elif dataset=="CPDB":
        print "Loading CPDB dataset"        
        g_it=load_graph_datasets.load_graphs_CPDB()
    elif dataset=="AIDS":
        print "Loading AIDS dataset"        
        g_it=load_graph_datasets.load_graphs_AIDS()
    elif dataset=="NCI1":
        print "Loading NCI1 dataset"        
        g_it=load_graph_datasets.load_graphs_NCI1()
    elif dataset=="NCI109":
        print "Loading NCI109 dataset"        
        g_it=load_graph_datasets.load_graphs_NCI109()
    else:
        print "Unknown dataset name"
     

    if kernel=="WL":
        print "Lambda ignored"
        print "Using WL fast subtree kernel"
        ODDkernel=WLGraphKernel(r=max_radius,normalization=normalization)
    elif kernel=="ODDST":
        print "Using ST kernel"
        ODDkernel=ODDSTGraphKernel(r=max_radius,l=la,normalization=normalization)#,ntype=ntype)
    elif kernel=="ODDSTP":
        print "Using ST+ kernel"
        ODDkernel=ODDSTPGraphKernel(r=max_radius,l=la,normalization=normalization,ntype=ntype,nsplit=nsplit)
    elif kernel=="NSPDK":
        print "Using NSPDK kernel, lambda parameter interpreted as d"
        ODDkernel=NSPDKGraphKernel(r=max_radius,d=int(la),normalization=normalization)
    elif kernel=="ODDSTC":
        print "Using ST kernel with contexts"
        ODDkernel=ODDSTCGraphKernel(r=max_radius,l=la,normalization=normalization,version=version,ntype=ntype,nsplit=nsplit)
    elif kernel=="ODDSTPC":
        print "Using ST+ kernel with contexts"
        ODDkernel=ODDSTPCGraphKernel(r=max_radius,l=la,normalization=normalization,version=version,ntype=ntype,nsplit=nsplit)
    elif kernel=="WLC":
        print "Lambda ignored"
        print "Using WL fast subtree kernel with contexts"
        ODDkernel=WLCGraphKernel(r=max_radius,normalization=normalization,version=version)
    elif kernel=="WLDDK":
        print "Using ST base kernel with WL kernel"
        ODDkernel=WLDDKGraphKernel(r=max_radius,h=iterations,l=la,normalization=normalization)
    elif kernel=="WLNSK":
        print "Using NS base kernel with WL kernel"
        ODDkernel=WLNSKGraphKernel(r=max_radius,h=iterations,l=la,normalization=normalization)
    else:
        print "Unrecognized kernel"
       
    GM=ODDkernel.computeKernelMatrixTrain(g_it.graphs) #Parallel ,njobs

    #print GM
#    GMsvm=[]    
#    for i in xrange(len(GM)):
#        GMsvm.append([])
#        GMsvm[i]=[i+1]
#        GMsvm[i].extend(GM[i])
#    #print GMsvm
#    from sklearn import datasets
#    print "Saving Gram matrix"
#    #datasets.dump_svmlight_file(GMsvm,g_it.target, name+".svmlight")
#    datasets.dump_svmlight_file(np.array(GMsvm),g_it.target, name+".svmlight")
#    #Test manual dump

    # tt = time.time()
    # tc = time.clock()

    print "Saving Gram matrix"
    #output=open(name+".svmlight","w")
    #for i in xrange(len(GM)):
    #    output.write(str(g_it.target[i])+" 0:"+str(i+1)+" ")
    #    for j in range(len(GM[i])):
    #        output.write(str(j+1)+":"+str(GM[i][j])+" ")
    #    output.write("\n")

    #output.close()

    # print GMsvm
    # from sklearn import datasets
    from sklearn.datasets import dump_svmlight_file
    dump_svmlight_file(GM, g_it.target, name+".svmlight")

    # print(time.time()-tt, time.clock()-tc)
    # print GM

