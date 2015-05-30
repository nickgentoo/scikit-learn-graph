__author__ = "Nicolo' Navarin"

import sys
from skgraph.kernel.ODDSTGraphKernel import ODDSTGraphKernel
from skgraph.datasets import load_graph_datasets

if __name__=='__main__':
    if len(sys.argv)<6:
        sys.exit("python ODDKernel_example.py dataset_name max_radius lambda_tree njobs filename")
    dataset=sys.argv[1]
    max_radius=int(sys.argv[2])
    la=float(sys.argv[3])
    #hashs=int(sys.argv[3])
    njobs=int(int(sys.argv[4]))
    name=str(sys.argv[5])


    #g_it=datasets.load_graphs_bursi()
    if dataset=="enzymes":
        print "Loading enzymes dataset"        
        g_it=load_graph_datasets.load_graphs_enzymes()
    elif dataset=="proteins":
        print "Loading proteins dataset"        
        g_it=load_graph_datasets.load_graphs_proteins()       
    elif dataset=="synthetic":
        print "Loading synthetic dataset"        
        g_it=load_graph_datasets.load_graphs_synthetic()   
    else:
        sys.exit( "ERROR: no dataset named "+dataset)
    print "labels",g_it.labels
    print "veclabels",g_it.veclabels

    ODDkernel=ODDSTGraphKernel(r=max_radius,l=la)
    #print ODDkernel.kernelFunctionFast(g_it.graphs[0],g_it.graphs[1])    
    GM=ODDkernel.computeKernelMatrixTrain(g_it.graphs) #Parallel ,njobs
    #GM=ODDkernel.computeKernelMatrixParallel(g_it.graphs,njobs) #Parallel ,njobs
    #GM=ODDkernel.computeKernelMatrixFeatureVectorParallel(g_it.graphs,njobs) #Parallel ,njobs

    #GM=ODDkernel.computeKernelMatrixFeatureVectorParallel([g_it.graphs[i] for i in range(21)],njobs) #Parallel ,njobs
    
    GMsvm=[]    
    for i in range(len(GM)):
        GMsvm.append([])
        GMsvm[i]=[i+1]
        GMsvm[i].extend(GM[i])
    from sklearn import datasets
    #datasets.dump_svmlight_file(GMsvm,g_it.target, name+".svmlight")
    datasets.dump_svmlight_file(GMsvm,g_it.target, name+".svmlight")
    #print GM
