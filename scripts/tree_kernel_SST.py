from skgraph.datasets import load_tree_datasets
from skgraph.tree.tree_kernels_SSTprod_new import SSTprodKernel
import sys
if __name__=='__main__':
    if len(sys.argv)<1:
        sys.exit("python tree_kernel_SST.py lambda njobs filename")
    max_radius=3
    la=float(sys.argv[1])
    njobs=int(sys.argv[2])
    name=str(sys.argv[3])
g_it=load_tree_datasets.load_trees_CAvsCM()

SSTkernel=SSTprodKernel(l=la)

GM=SSTkernel.computeKernelMatrixTrain([g_it.graphs[i] for i in range(21)]) #Parallel ,njobs
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



 
