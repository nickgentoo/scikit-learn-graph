import tree
import tree_kernels
import sys

if len(sys.argv)<3:
    sys.exit("python prova2.py inputDataset kernelParameter(Lambda)")

dat = tree.TreeDatasetPrologFormat()
dat.loadFromFile(sys.argv[1])
kfunc = tree_kernels.KernelSST(float(sys.argv[2]))
#kmatrix = tree_kernels.KernelMatrixLibsvm.compute(dat, kfunc) 
kmatrix = tree_kernels.KernelMatrix.compute(dat, kfunc) 
kmatrix.Print(sys.argv[1] + ".kmatrix")
