# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:52:04 2015

Copyright 2015 Riccardo Tesselli,Nicolo' Navarin

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

import pylab
import networkx as nx
from collections import deque
from copy import deepcopy
def drawWeightedGraph(G):
    """
    Function that draws a networkx graph with weight attributes
    @type G: networkx graph
    @param G: The graph to draw
    """
    weights=[d['weight'] for n,d in G.nodes(data=True)]
    pos1=nx.graphviz_layout(G)
    nx.draw_networkx(G,node_color=weights,pos=pos1,node_size=500)
    pylab.show()

def drawGraph(G,indexroot=-1):
    """
    Function that draws a networkx graph
    @type G: networkx graph
    @param G: The graph to draw
    
    @type indexroot: integer number
    @param indexroot: the index of the graph's root (if needed). -1 means no root
    """
    labels=dict((n,d['label']) for n,d in G.nodes(data=True))
    labels={k:str(k)+':'+v for (k,v) in labels.items()}
    pos1=nx.graphviz_layout(G)
    nx.draw_networkx(G,labels=labels,pos=pos1,node_size=500)
    if indexroot>-1:
        nx.draw_networkx_nodes(G,pos1, nodelist=[indexroot], node_color='b', node_size=500)
    pylab.show()

def generateDAG(G,index_start_node,height):
    """
    Given a Networkx graph G, an integer node index and an integer height returns the correspondent Decompositional DAG
    @type G: networkx graph
    @param G: graph to generate the dag from
    
    @type index_start_node: integer number
    @param index_start_node: root's id
    
    @type height: integer number
    @param height: max DAG's depth
    
    @rtype: tuple
    @return: built DAG and its maxheight
    """
    Dict_distance_from_root={} #Create map node distance from root
    Dict_distance_from_root[index_start_node]=0
    Dict_already_explored={} #We use a map instead of a vector since the limited depth breadth first visit can visit much fewer vertices than there are vertices in total
    Dict_already_explored[index_start_node]=True
    Deque_queue=deque([index_start_node]) #Initialize queue
    Dict_labels=dict((n,d['label']) for n,d in G.nodes(data=True))
    #print G.nodes(data=True)[1]
    if 'veclabel' in G.nodes(data=True)[0][1]:
        #print "veclabels present"
        Dict_veclabels=dict((n,d['veclabel']) for n,d in G.nodes(data=True))
    else:
        pass
        #print "no vectlabels detected"
    DAG=nx.DiGraph() #Directed Graph
    if 'veclabel' in G.nodes(data=True)[0][1]:
        DAG.add_node(index_start_node, depth=0,label=Dict_labels[index_start_node],veclabel=Dict_veclabels[index_start_node])

    else:       
        DAG.add_node(index_start_node, depth=0,label=Dict_labels[index_start_node])
    DAG.graph['root']=index_start_node
    maxLevel=0
    while Deque_queue: #while the queue is not empty
        u=Deque_queue[0]
        for v in nx.all_neighbors(G,u):
            if Dict_already_explored.get(v) and Dict_distance_from_root[u]+1 <= height and Dict_distance_from_root[v] > Dict_distance_from_root[u]:
                DAG.add_edge(u,v)#Diamond structure detected
                
            if Dict_already_explored.get(v) is None:
                if Dict_distance_from_root[u]+1 <=height:#if next step is permitted
                    Dict_distance_from_root[v]=Dict_distance_from_root[u]+1 #update values
                    if maxLevel<Dict_distance_from_root[v]:
                        maxLevel=Dict_distance_from_root[v]
                    Dict_already_explored[v]=True
                    Deque_queue.append(v)
                    if 'veclabel' in G.nodes(data=True)[0][1]:
                        DAG.add_node(v, depth=Dict_distance_from_root[v],label=Dict_labels[v],veclabel=Dict_veclabels[v])

                    else:
                        DAG.add_node(v, depth=Dict_distance_from_root[v],label=Dict_labels[v])
                    DAG.add_edge(u,v)
                    
        Deque_queue.popleft() #current node has been processed, process next one
    return (DAG,maxLevel)
    
def generateDAGOrdered(G,index_start_node,height):
    """
    Given a Networkx graph G, an integer node index and an integer height returns the correspondent Decompositional DAG
    @type G: networkx graph
    @param G: graph to generate the dag from
    
    @type index_start_node: integer number
    @param index_start_node: root's id
    
    @type height: integer number
    @param height: max DAG's depth
    
    @rtype: tuple
    @return: built DAG and its maxheight
    """
    Dict_distance_from_root={} #Create map node distance from root
    Dict_distance_from_root[index_start_node]=0
    Dict_already_explored={} #We use a map instead of a vector since the limited depth breadth first visit can visit much fewer vertices than there are vertices in total
    Dict_already_explored[index_start_node]=True
    Deque_queue=deque([index_start_node]) #Initialize queue
    Dict_labels=dict((n,d['label']) for n,d in G.nodes(data=True))
    #print G.nodes(data=True)[1]
    if 'veclabel' in G.nodes(data=True)[0][1]:
        #print "veclabels present"
        Dict_veclabels=dict((n,d['veclabel']) for n,d in G.nodes(data=True))
    if 'childrenOrder' in G.nodes(data=True)[0][1]:
        #print "veclabels present"
        Dict_childrenOrder=dict((n,deepcopy(d['childrenOrder'])) for n,d in G.nodes(data=True))
        #print "no vectlabels detected"
    DAG=nx.DiGraph() #Directed Graph
    if 'veclabel' in G.nodes(data=True)[0][1]:
        DAG.add_node(index_start_node, depth=0,label=Dict_labels[index_start_node],veclabel=Dict_veclabels[index_start_node])

    else:       
        DAG.add_node(index_start_node, depth=0,label=Dict_labels[index_start_node])
    
    if 'childrenOrder' in G.nodes(data=True)[0][1]:
        DAG.node[index_start_node]['childrenOrder']=Dict_childrenOrder[index_start_node]
    DAG.graph['root']=index_start_node
    maxLevel=0
    while Deque_queue: #while the queue is not empty
        u=Deque_queue[0]
        #print 'processing ',u, DAG.node[u]['childrenOrder'], G[u]
        for v in G.successors(u):
            edge_added=False
            if Dict_already_explored.get(v) and Dict_distance_from_root[u]+1 <= height and Dict_distance_from_root[v] > Dict_distance_from_root[u]:
                DAG.add_edge(u,v)#Diamond structure detected
                edge_added=True
                pass

                
            if Dict_already_explored.get(v) is None:
                if Dict_distance_from_root[u]+1 <=height:#if next step is permitted
                    Dict_distance_from_root[v]=Dict_distance_from_root[u]+1 #update values
                    if maxLevel<Dict_distance_from_root[v]:
                        maxLevel=Dict_distance_from_root[v]
                    Dict_already_explored[v]=True
                    Deque_queue.append(v)

                    DAG.add_node(v, depth=Dict_distance_from_root[v],label=Dict_labels[v])
                    
                    #if leaf node, childrenOrder must be empty
                    if 'childrenOrder' in G.nodes(data=True)[0][1]:
                        if Dict_distance_from_root[u]+1 <height:
                            DAG.node[v]['childrenOrder']=Dict_childrenOrder[v]
                        else:
                            DAG.node[v]['childrenOrder']=Dict_childrenOrder[v]

                    DAG.add_edge(u,v)
                    edge_added=True

            if not edge_added:#rimuovo v da childrenOrder
                if v in DAG.node[u]['childrenOrder']:
                    DAG.node[u]['childrenOrder'].remove(v)
                    
                        
        Deque_queue.popleft() #current node has been processed, process next one
    for i in DAG.nodes():
        assert len(DAG.node[i]['childrenOrder'])==DAG.out_degree(i)
    return (DAG,maxLevel)
    
    
def orderDAGvertices(D):
    """
    Given a Networkx graph G, an integer node index and an integer height returns the correspondent Decompositional DAG
    @type G: networkx graph
    @param G: graph to generate the dag from
    """
    setOrder(D,D.graph['root'],sep='|')
    
def setOrder(T, nodeID, sep='|'):
       return setOrderNoVeclabels(T, nodeID,sep)


def setOrderNoVeclabels(T, nodeID, sep):
        """
        The method computes an identifier of the node based on
        1) the label of the node self
        2) the hash values of the children of self
        For each visited node the hash value is stored into the attribute subtreeId
        The label and the identifiers of the children nodes are separated by the char 'sep'
        """
        #print "labels",labels
        #print "sep",sep

        if 'orderString' in T.node[nodeID]:
            return T.node[nodeID]['orderString']
        stri = str(T.node[nodeID]['label'])

        #append veclabels if available
        #print str(T.node[nodeID]['veclabel'])
        #print stri
        if stri.find(sep) != -1:
            print "ERROR: identifier " + sep + "used in label. Please set it with setHashSep(newsep)"
        succ_labels=[]
        if len(T.successors(nodeID))>0:
            stri +=sep+str(len(T.successors(nodeID)))
        for c in T.successors(nodeID):
            tup=(setOrderNoVeclabels(T,c,sep),c)
            succ_labels.append(tup)
            succ_labels.sort(cmp = lambda x, y: cmp(x[0], y[0]))
        children=[]
        for l in  succ_labels:
            stri += sep + str(l[0])
            children.append(l[1])
        T.node[nodeID]['orderString'] = stri
        T.node[nodeID]['childrenOrder']= children
        #debug
        #print T.node[nodeID]['orderString']
        #print T.node[nodeID]['childrenOrder']
        return T.node[nodeID]['orderString']