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
from scipy.sparse import csr_matrix
from ioskgraph import load_target
from ..graph import instance_to_graph
from sklearn.datasets.base import Bunch
from obabel import obabel_to_eden
def convert_to_sparse_matrix(km):
    # translate dictionary to Compressed Sparse Row matrix
        if len(km) == 0:
            raise Exception('ERROR: something went wrong, empty feature_dict. Perhaps wrong data format, i.e. do nodes have the "viewpoint" attribute?')
        row, col, data = [], [], []
        ne = len(km)
        for i in range(ne):
            for j in range(ne):
                if (km[i][j]!=0):
                    row.append( i )
                    col.append( j )
                    data.append(km[i][j])
        print len(row),len(col),len(data)
          
        X = csr_matrix( (data,(row,col)), shape = (ne, ne))

        return X
def load_graphs_GDD():
    """Load the GDD graph dataset for graph classification..

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/GDD/GDD_labels.txt'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/GDD/graphs.gspan'
    _target=load_target(input_target_url)
    g_it=instance_to_graph(input = input_data_url)
    gra=[i for i in g_it]
    print 'Loaded GDD graph dataset for graph classification.'
    print len(gra),'graphs.'    
    return Bunch(graphs=gra,
    target=_target,
    labels=True,
    veclabels=False)

def load_graphs_MUTAG():
    """Load the MUTAG graph dataset for graph classification..

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/MUTAG/mutag_188_target.txt'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/MUTAG/mutag_188_data.can'
    _target=load_target(input_target_url)
    g_it=obabel_to_eden(input = input_data_url,file_type ='smi')

    gra=[i for i in g_it]
    print 'Loaded MUTAG graph dataset for graph classification.'
    print len(gra),'graphs.'
    return Bunch(graphs=gra,
    target=_target,
    labels=True,
    veclabels=False)

def load_graphs_CPDB():
    """Load the CPDB graph dataset for graph classification..

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/CPDB/mutagen_labels.tab'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/CPDB/mutagen_smile.can'
    _target=load_target(input_target_url)
    g_it=obabel_to_eden(input = input_data_url,file_type ='smi')

    gra=[i for i in g_it]
    print 'Loaded CPDB graph dataset for graph classification.'
    print len(gra),'graphs.'
    return Bunch(graphs=gra,
    target=_target,
    labels=True,
    veclabels=False)
    
def load_graphs_AIDS():
    """Load the AIDS graph dataset for graph classification..

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/AIDS/CAvsCM.y'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/AIDS/CAvsCM.can'
    _target=load_target(input_target_url)
    g_it=obabel_to_eden(input = input_data_url,file_type ='smi')

    gra=[i for i in g_it]
    print 'Loaded AIDS graph dataset for graph classification.'
    print len(gra),'graphs.'
    return Bunch(graphs=gra,
    target=_target,
    labels=True,
    veclabels=False)

def load_graphs_NCI1():
    """Load the NCI1 graph dataset for graph classification..

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/NCI1/NCI1_labels.txt'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/NCI1/NCI1_graphs.gspan'
    _target=load_target(input_target_url)
    g_it=instance_to_graph(input = input_data_url)

    print 'Loaded NCI1 graph dataset for graph classification.'
    return Bunch(graphs=[i for i in g_it],
    target=_target,
    labels=True,
    veclabels=False)
        
def load_graphs_bursi():
    """Load the Bursi graph dataset for graph classification..

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.bioinf.uni-freiburg.de/~costa/bursi.target'
    input_data_url='http://www.bioinf.uni-freiburg.de/~costa/bursi.gspan'
    _target=load_target(input_target_url)
    g_it=instance_to_graph(input = input_data_url)

    print 'Loaded Bursi graph dataset for graph classification.'
    return Bunch(graphs=[i for i in g_it],
    target=_target,
    labels=True,
    veclabels=False)


def load_graphs_enzymes():
    """Load the ENZYMES graph dataset for (multiclass) graph classification from:
    Schomburg, I., Chang, A., Ebeling, C., Gremse, M., Heldt, C., Huhn, G., & Schomburg, D. (2004).
    BRENDA, the enzyme database: updates and major new developments.
    Nucleic Acids Research, 32, D431–D433. doi:10.1093/nar/gkh081

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/ENZYMES.labels'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/ENZYMES.gspan'
    #input_target_url='datasets/ENZYMES.labels'  
    #input_data_url='datasets/ENZYMES.gspan'
    
    _target=load_target(input_target_url)
    g_it=instance_to_graph(input = input_data_url) #url
    #return Bunch(data=flat_data,
    #            target=target.astype(np.int),
    #           target_names=np.arange(10),
    #            images=images,
    #            DESCR=descr)
    print 'Loaded ENZYMES graph dataset for (multiclass) graph classification from:'
    print 'Schomburg, I., Chang, A., Ebeling, C., Gremse, M., Heldt, C., Huhn, G., & Schomburg, D. (2004).'
    print 'BRENDA, the enzyme database: updates and major new developments.'
    print 'Nucleic Acids Research, 32, D431–D433. doi:10.1093/nar/gkh081'
    return Bunch(graphs=[i for i in g_it],
    target=_target,
    labels=True,
    veclabels=True)
    
def load_graphs_proteins():
    """Load the PROTEINS graph dataset for graph classification from:
        Dobson, P. D., & Doig, A. J. (2003)
        Distinguishing enzyme structures from non-enzymes without alignments.
        Journal of Molecular Biology, 330, 771–783. doi:10.1016/S0022-2836(03)00628-4

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/PROTEINS.labels'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/PROTEINS.gspan'
    
    _target=load_target(input_target_url)
    g_it=instance_to_graph(input = input_data_url) #url

    print 'Loaded PROTEINS graph dataset for graph classification from:'    
    print 'Dobson, P. D., & Doig, A. J. (2003)'
    print 'Distinguishing enzyme structures from non-enzymes without alignments.'
    print 'Journal of Molecular Biology, 330, 771–783. doi:10.1016/S0022-2836(03)00628-4'
    return Bunch(graphs=[i for i in g_it],
    target=_target,
    labels=True,
    veclabels=True)
    
def load_graphs_synthetic():
    """Load the SYNTHETIC graph dataset for graph classification from:      
        Feragen, A., Kasenburg, N., Petersen, J., de Bruijne, M., & Borgwardt, K. M. (2013)
        Scalable kernels for graphs with continuous attributes.
        In Neural Information Processing Systems (NIPS) 2013 (pp. 216–224).
        Retrieved from http://papers.nips.cc/paper/5155-scalable-kernels-for-graphs-with-continuous-attributes.pdf
    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/SYNTHETICnew.labels'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/SYNTHETICnew.gspan'
    #input_target_url='datasets/ENZYMES.labels'  
    #input_data_url='datasets/ENZYMES.gspan'
    
    _target=load_target(input_target_url)
    g_it=instance_to_graph(input = input_data_url) #url
    #return Bunch(data=flat_data,
    #            target=target.astype(np.int),
    #           target_names=np.arange(10),
    #            images=images,
    #            DESCR=descr)
    g=[i for i in g_it]
    for i in g:
        for n in i.nodes():
            i.node[n]['label']=str(i.degree(n))
    
    print 'Loaded SYNTHETIC graph dataset for graph classification from:'    
    print 'Feragen, A., Kasenburg, N., Petersen, J., de Bruijne, M., & Borgwardt, K. M. (2013)'
    print 'Scalable kernels for graphs with continuous attributes.'
    print 'In Neural Information Processing Systems (NIPS) 2013 (pp. 216–224).'
    return Bunch(graphs=g,
    target=_target,
    labels=True,
    veclabels=True)
    
def load_graphs_BZR():
    """Load the BZR graph dataset for graph classification from:
        Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph 
        Kernels from Propagated Information. Under review at MLJ.

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/BZR_graph_labels.txt'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/BZR.gspan'
    #input_target_url='datasets/ENZYMES.labels'  
    #input_data_url='datasets/ENZYMES.gspan'
    
    _target=load_target(input_target_url)
    g_it=instance_to_graph(input = input_data_url) #url
    #return Bunch(data=flat_data,
    #            target=target.astype(np.int),
    #           target_names=np.arange(10),
    #            images=images,
    #            DESCR=descr)
    print 'Loaded BZR graph dataset for  graph classification from:'
    print 'Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph' 
    print 'Kernels from Propagated Information. MLJ 2015.'
    return Bunch(graphs=[i for i in g_it],
    target=_target,
    labels=True,
    veclabels=True)
    
def load_graphs_COX2():
    """Load the COX2 graph dataset for graph classification from:
        Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph 
        Kernels from Propagated Information. Under review at MLJ.

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/COX2_graph_labels.txt'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/COX2.gspan'
    #input_target_url='datasets/ENZYMES.labels'  
    #input_data_url='datasets/ENZYMES.gspan'
    
    _target=load_target(input_target_url)
    g_it=instance_to_graph(input = input_data_url) #url
    #return Bunch(data=flat_data,
    #            target=target.astype(np.int),
    #           target_names=np.arange(10),
    #            images=images,
    #            DESCR=descr)
    print 'Loaded COX2 graph dataset for  graph classification from:'
    print 'Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph' 
    print 'Kernels from Propagated Information. MLJ 2015.'
    return Bunch(graphs=[i for i in g_it],
    target=_target,
    labels=True,
    veclabels=True)
    
def load_graphs_DHFR():
    """Load the DHFR graph dataset for graph classification from:
        Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph 
        Kernels from Propagated Information. Under review at MLJ.

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/DHFR_graph_labels.txt'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/DHFR.gspan'
    #input_target_url='datasets/ENZYMES.labels'  
    #input_data_url='datasets/ENZYMES.gspan'
    
    _target=load_target(input_target_url)
    g_it=instance_to_graph(input = input_data_url) #url
    #return Bunch(data=flat_data,
    #            target=target.astype(np.int),
    #           target_names=np.arange(10),
    #            images=images,
    #            DESCR=descr)
    print 'Loaded DHFR graph dataset for  graph classification from:'
    print 'Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph' 
    print 'Kernels from Propagated Information. MLJ 2015.'
    return Bunch(graphs=[i for i in g_it],
    target=_target,
    labels=True,
    veclabels=True)
    
def load_graphs_PROTEINS_full():
    """Load the PROTEINS_full graph dataset for graph classification from:
        Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph 
        Kernels from Propagated Information. Under review at MLJ.

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/PROTEINS_full_graph_labels.txt'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/PROTEINS_full.gspan'
    #input_target_url='datasets/ENZYMES.labels'  
    #input_data_url='datasets/ENZYMES.gspan'
    
    _target=load_target(input_target_url)
    g_it=instance_to_graph(input = input_data_url) #url
    #return Bunch(data=flat_data,
    #            target=target.astype(np.int),
    #           target_names=np.arange(10),
    #            images=images,
    #            DESCR=descr)
    print 'Loaded PROTEINS_full graph dataset for  graph classification from:'
    print 'Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph' 
    print 'Kernels from Propagated Information. MLJ 2015.'
    return Bunch(graphs=[i for i in g_it],
    target=_target,
    labels=True,
    veclabels=True)