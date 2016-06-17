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

from numpy.linalg import norm



def instance_to_graph(input = None,dict_labels={}, counter=[1]):
    """
    Function that reads a graph dataset encoded in gspan format from an inpout stream.

    """
    import gspan
    return gspan.gspan_to_eden(input,dict_labels,counter)


#def instance_to_graph(input = None, input_type = 'file', tool = None, options = dict()):
#    """
#    tool= "gspan" "node_link_data" "sequence" "obabel"
#    """
#    return convert.instance_to_eden(input, input_type, tool, options)


##utility functions
def setHashSubtreeIdentifierBigDAG(T, nodeID, sep='|',labels=True,veclabels=True):
        """
        The method computes an identifier of the node based on
        1) the label of the node self
        2) the hash values of the children of self
        For each visited node the hash value is stored into the attribute subtreeId
        The label and the identifiers of the children nodes are separated by the char 'sep'
        """
        #print "labels",labels
        #TODO tipo di ordine
        setOrder(T, nodeID, labels=labels,veclabels=veclabels)
        if 'subtreeID' in T.node[nodeID]:
            return T.node[nodeID]['subtreeID']
        stri = str(T.node[nodeID]['label'])+str(T.node[nodeID]['veclabel'])
        if stri.find(sep) != -1:
            print "ERROR: identifier " + sep + "used in label. Please set it with setHashSep(newsep)"
        for c in T.node[nodeID]['childrenOrder']:#T.successors(nodeID):
            stri += sep + setHashSubtreeIdentifierBigDAG(T,c,sep,labels,veclabels)
        #print stri
        T.node[nodeID]['subtreeID'] = str(stri)#hash(
        return T.node[nodeID]['subtreeID']

## GRAPH FUNCTIONS
def setOrder(T, nodeID, sep='|',labels=True,veclabels=True,order="norm"):
    if veclabels:
        return setOrderVeclabels(T, nodeID,sep,labels,order)
    else:
       return setOrderNoVeclabels(T, nodeID,sep,labels)


def setOrderVeclabels(T, nodeID, sep,labels,order):
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
   
        #order according kernel between veclabels and the one of the father
 
            
        #print str(T.node[nodeID]['veclabel'])
        #print stri
        if stri.find(sep) != -1:
            print "ERROR: identifier " + sep + "used in label. Please set it with setHashSep(newsep)"
        succ_labels=[]
        if len(T.successors(nodeID))>0:
            stri +=sep+str(len(T.successors(nodeID)))
        for c in T.successors(nodeID):
            if order=='gaussian':
                dist=gaussianKernel(T.node[nodeID]['veclabel'],T.node[c]['veclabel'],1.0/len(T.node[c]['veclabel']))#self.beta)
            elif order=='norm':
                dist=norm(T.node[nodeID]['veclabel'])
            else:
                print "no ordering specified"
            tup=([setOrderVeclabels(T,c,sep,labels,order),dist],c)
            succ_labels.append(tup)
        #print "before sorting",succ_labels
        succ_labels.sort(key=lambda x:(x[0][0],x[0][1]))#cmp = lambda x, y: cmp(x[0], y[0])
        #print "after sorting",succ_labels
        children=[]
        for l in  succ_labels:
            stri += sep + str(l[0][0])
            children.append(l[1])
        T.node[nodeID]['orderString'] = stri
        #print "order string", stri
        T.node[nodeID]['childrenOrder']= children
        #print "order children", children

        #debug
        #print T.node[nodeID]['orderString']
        #print T.node[nodeID]['childrenOrder']
        return T.node[nodeID]['orderString']

