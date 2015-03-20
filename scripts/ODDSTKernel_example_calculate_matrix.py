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
from skgraph.kernel.ODDSTGraphKernel import ODDSTGraphKernel
from skgraph.datasets import load_graph_datasets


if __name__=='__main__':
    if len(sys.argv)<1:
        sys.exit("python ODDKernel_example.py filename")
    max_radius=3
    la=1
    #hashs=int(sys.argv[3])
    njobs=1
    name=str(sys.argv[1])

     
    g_it=load_graph_datasets.load_graphs_bursi()


    ODDkernel=ODDSTGraphKernel(r=max_radius,l=la)
    GM=ODDkernel.computeKernelMatrixTrain([g_it.graphs[i] for i in range(21)]) #Parallel ,njobs
    GMsvm=[]    
    for i in range(len(GM)):
        GMsvm.append([])
        GMsvm[i]=[i+1]
        GMsvm[i].extend(GM[i])
    from sklearn import datasets
    print "Saving Gram matrix"
    #datasets.dump_svmlight_file(GMsvm,g_it.target, name+".svmlight")
    datasets.dump_svmlight_file(GMsvm,[g_it.target[i] for i in range(21)], name+".svmlight")
    #print GM