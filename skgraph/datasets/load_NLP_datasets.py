# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:52:16 2015

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
import networkx as nx
import sys
import ioskgraph

def load_graphs_paraphrase_train():
     print "Loading Microsoft Paraphrase dataset (training set)" 
     G_it = readNLPGraphDataset('http://www.math.unipd.it/~nnavarin/datasets/Paraphrase/train.constREL.graph')
     for g in G_it.graphs:
         g.graph['ordered']=True
         for v in g.nodes():
             if g.node[v]['label'].endswith('-REL'):
                 #print g.node[v]['label']
                 g.node[v]['viewpoint']=True
             else:
                 g.node[v]['viewpoint']=False
     return G_it

def load_graphs_paraphrase_test():
     print "Loading Microsoft Paraphrase dataset (test set)" 
     G_it = readNLPGraphDataset('http://www.math.unipd.it/~nnavarin/datasets/Paraphrase/test.constREL.graph')
     for g in G_it.graphs:
         g.graph['ordered']=True
         for v in g.nodes():
             if g.node[v]['label'].endswith('-REL'):
                 #print g.node[v]['label']
                 g.node[v]['viewpoint']=True
             else:
                 g.node[v]['viewpoint']=False
     return G_it


def readNLPGraphDataset(graphDatasetFile):
    
    Glist = []
    targetList = []
    indList = []
    s = ""
    #Y = [y.strip() for y in ioskgraph.read(graphDatasetFile) if y]
    #print Y
    #with open(inputfile, "r") as f:
        #for line in f.readlines():

    for line in ioskgraph.read(graphDatasetFile):
        #print line
        if line.strip():

            if line[0:3]=="<BG":
                s = line+'\n'
            elif line[0:4] == "<EG>":
                s += line
                G = fromString(s)
                Glist.append(G)
                targetList.append(int(G.graph['target']))
                indList.append(int(G.graph['index'])-1)
            else:
                s += line+'\n'
    for i in xrange(len(indList)):
        assert (indList[i]==i)
    print len(Glist), "examples in the dataset"    
    #print "the list of graph is in Glist array"
    #print "the list of target values is in targetList array"
    #print "if you want to access the x-th example (1<=x<=len(dataset)) do Glist[indList[x]] and targetList[indList[x]]"
    from sklearn.datasets.base import Bunch
    return Bunch(graphs=Glist,
    target=targetList,
    labels=True,
    veclabels=False)


def nodeAttributeToString(h):
    return ''.join(list(h['label']))

def edgeAttributeToString(h):
    return h['childindex']

def toString(G):
    s = ""
    for k in G.graph.keys():
        s += " " + k + "=" + str(G.graph[k])
    s = "<BG:gio>" + s + "\n"
    l = []
    nnodes = nx.number_of_nodes(G)
    i = 0
    for n in sorted([ (int(x[0]), x[1]['ind'],nodeAttributeToString(x[1])) for x in G.nodes_iter(data=True)]):
        i+=1
        s += "N %s %s\n"%(n[1],n[2])
    if not i==nnodes:
        print "diversi: " + i + " " + nnodes
    nedges = nx.number_of_edges(G)
    i = 0
    #for e in sorted([(min(int(x[0]), int(x[1])), max(int(x[0]),int(x[1])), x[2]) for x in G.edges_iter(data=True)]):#undirected
    for e in sorted([(int(x[0]), int(x[1]), x[2]) for x in G.edges_iter(data=True)]):#directed
        i+=1
        s += "E %s %s %s\n" %(e[0], e[1], edgeAttributeToString(e[2]))
    if not nedges==i:
        print "archi diversi: " + i + " " + nedges
    s += "<EG>\n"
    return s


def fromString(s):
    line = s.split("\n")
    style = line[0][line[0].find("<BG"):line[0].find(">")]
    G = nx.DiGraph()
    attr = line[0][line[0].find(">")+2:].rstrip()
    if len(attr)>0:
        for k in attr.split(" "):
            kk,v = k.split("=")
            G.graph[kk] = v
    row = 1
    numrows = len(line)
    while row < numrows and line[row][0]=="N":
        r = line[row].rstrip().split(" ")
        if len(r) < 2:
            sys.exit("ERROR: graph node index not found on line " + line[row])
        G.add_node(int(r[1])+1)
        G.node[int(r[1])+1]['ind'] = int(r[1])+1
        G.node[int(r[1])+1]['viewpoint'] = True
        G.node[int(r[1])+1]['childrenOrder']=[]
        if len(r) > 2:
            G.node[int(r[1])+1]['label'] = ("".join(r[2:]).encode('utf-8'))
        row += 1
    while row < numrows and line[row][0]=="E":
        r = line[row].rstrip().split(" ")
        if len(r) < 3:
            sys.exit("ERROR: graph edge index not found on line " + line[row])
        G.add_edge(int(r[1])+1, int(r[2])+1)
        if len(r) == 3:
            print "Error unordered edge!"
        if len(r) > 3:
            index=int(r[3])-2
            #extend list if it is not long enough for the index
            G.node[int(r[1])+1]['childrenOrder'].extend([-1 for i in xrange(len(G.node[int(r[1])+1]['childrenOrder']),index+1)])
            G.node[int(r[1])+1]['childrenOrder'][index]=int(r[2])+1
            #print G.node[int(r[1])]['childrenOrder']
            #childrenOrder sould be correct
            
            G.edge[int(r[1])+1][int(r[2])+1]['childindex'] = r[3]
            #G.edge[int(r[1])][r[2]]['label'] = r[3:]

        row += 1
    #integrity check
    for i in G.nodes():
        #print i,DAG.node[i]['childrenOrder'],DAG.out_degree(i)
        assert len(G.node[i]['childrenOrder'])==G.out_degree(i)
    if not line[row]=="<EG>":
        sys.error("ERROR: cannot find expected <EG> marker at the end of the graph")
    return G

