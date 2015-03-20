#!/usr/bin/python

import sys
import tree
from ..skgraph.dependencies.pythontk import tree_kernels

l=0.5
m=0.2

dat = tree.TreeDatasetSvmlightFormat()
dat.loadFromFile(sys.argv[1])
kernelPT = tree_kernels.KernelPT(l, m, True)
km = tree_kernels.KernelMatrixLibsvm.compute(dat, kernelPT)
km.Print(sys.argv[1] + ".mykm.l" + str(l) + "-m" + str(m))
