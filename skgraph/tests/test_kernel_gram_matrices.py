__author__ = "Carlo Maria Massimo"
__date__ = "19/oct/2015"
__credits__ = ["Carlo Maria Massimo"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer = "Carlo Maria Massimo"
__email__ = "cmassim@gmail.com"
__status__ = "Production"

import nose
from nose.tools import assert_true
import numpy as np
import networkx as nx
from skgraph.kernel.ODDSTCGraphKernel import ODDSTCGraphKernel
from skgraph.kernel.ODDSTPCGraphKernel import ODDSTPCGraphKernel
from skgraph.kernel.ODDSTPGraphKernel import ODDSTPGraphKernel
from skgraph.kernel.WLDDKGraphKernel import WLDDKGraphKernel
from skgraph.kernel.WLNSKGraphKernel import WLNSKGraphKernel

t1 = nx.Graph()
t1.add_node(1, label='A')
t1.add_node(2, label='B')
t1.add_node(3, label='C')
t1.add_node(4, label='D')
t1.add_node(5, label='E')
t1.add_edge(1, 2)
t1.add_edge(1, 3)
t1.add_edge(3, 4)
t1.add_edge(4, 5)
nx.set_node_attributes(t1, 'viewpoint', True)

t2 = nx.Graph()
t2.add_node(1, label='A')
t2.add_node(2, label='B')
t2.add_node(3, label='C')
t2.add_node(4, label='D')
t2.add_node(5, label='E')
t2.add_edge(1, 2)
t2.add_edge(1, 3)
t2.add_edge(1, 4)
t2.add_edge(1, 5)
nx.set_node_attributes(t2, 'viewpoint', True)

t3 = nx.Graph()
t3.add_node(1, label='A')
t3.add_node(2, label='B')
t3.add_node(3, label='C')
t3.add_node(4, label='D')
t3.add_node(5, label='E')
t3.add_edge(1, 2)
t3.add_edge(1, 3)
t3.add_edge(2, 4)
t3.add_edge(3, 4)
t3.add_edge(4, 5)
nx.set_node_attributes(t3, 'viewpoint', True)

t4 = nx.Graph()
t4.add_node(1, label='A')
t4.add_node(2, label='B')
t4.add_node(3, label='C')
t4.add_node(4, label='D')
t4.add_node(5, label='E')
t4.add_edge(1, 2)
t4.add_edge(2, 3)
t4.add_edge(3, 4)
t4.add_edge(4, 5)
t4.add_edge(5, 1)
nx.set_node_attributes(t4, 'viewpoint', True)

toys = [t1, t2, t3, t4]

def test_oddstc_graph_kernel():
    calculated_gram = np.loadtxt("skgraph/tests/testdata/stc_toys.gram")

    k = ODDSTCGraphKernel(r = 3, l = 1, normalization = True)
    gram = k.computeKernelMatrixTrain(toys)

    assert_true(np.allclose(gram, calculated_gram, rtol = 1e-05, atol = 1e-08))

def test_oddstpc_graph_kernel():
    calculated_gram = np.loadtxt("skgraph/tests/testdata/stpc_toys.gram")

    k = ODDSTPCGraphKernel(r = 3, l = 1, normalization = True)
    gram = k.computeKernelMatrixTrain(toys)

    assert_true(np.allclose(gram, calculated_gram, rtol = 1e-05, atol = 1e-08))

def test_oddstp_graph_kernel():
    calculated_gram = np.loadtxt("skgraph/tests/testdata/stp_toys.gram")

    k = ODDSTPGraphKernel(r = 3, l = 1, normalization = True)
    gram = k.computeKernelMatrixTrain(toys)

    assert_true(np.allclose(gram, calculated_gram, rtol = 1e-05, atol = 1e-08))


def test_wlddk_graph_kernel():
    return
    calculated_gram = np.loadtxt("skgraph/tests/testdata/bursi.gspan.mtx.WLDDK")

    k = WLDDKGraphKernel(r = 3, l = 1, normalization = True)
    gram = k.computeKernelMatrixTrain(toys)

    assert_true(np.allclose(gram, calculated_gram, rtol = 1e-05, atol = 1e-08))


def test_wlnsk_graph_kernel():
    return
    calculated_gram = np.loadtxt("skgraph/tests/testdata/bursi.gspan.mtx.WLNSK")

    k = WLNSKGraphKernel(r = 3, l = 1, normalization = True)
    gram = k.computeKernelMatrixTrain(toys)

    assert_true(np.allclose(gram, calculated_gram, rtol = 1e-05, atol = 1e-08))

