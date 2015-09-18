# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 16:10:20 2015

@author: nick
"""

from skgraph.datasets import load_graph_datasets as datasets
dataset=datasets.load_graphs_COX2()
print len(dataset.graphs)
print len(dataset.target)
print dataset.graphs[0].node[1]['label']
print dataset.graphs[0].node[1]['veclabel']
dataset=datasets.load_graphs_BZR()
print len(dataset.graphs)
print len(dataset.target)
print dataset.graphs[0].node[1]['label']
print dataset.graphs[0].node[1]['veclabel']
dataset=datasets.load_graphs_DHFR()
print len(dataset.graphs)
print len(dataset.target)
print dataset.graphs[0].node[1]['label']
print dataset.graphs[0].node[1]['veclabel']
dataset=datasets.load_graphs_PROTEINS_full()
print len(dataset.graphs)
print len(dataset.target)
print dataset.graphs[0].node[1]['label']
print dataset.graphs[0].node[1]['veclabel']
dataset=datasets.load_graphs_bursi()
print len(dataset.graphs)
print len(dataset.target)
dataset=datasets.load_graphs_enzymes()
print len(dataset.graphs)
print len(dataset.target)
print dataset.graphs[1].node[1]['label']
print dataset.graphs[1].node[1]['veclabel']
dataset=datasets.load_graphs_proteins()
print len(dataset.graphs)
print len(dataset.target)
print dataset.graphs[1].node[1]['label']
print dataset.graphs[1].node[1]['veclabel']
dataset=datasets.load_graphs_synthetic()
print len(dataset.graphs)
print len(dataset.target)
print dataset.graphs[1].node[1]['label']
print dataset.graphs[1].node[1]['veclabel']
