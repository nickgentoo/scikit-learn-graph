# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:55:12 2015
This file is (slightly modified) from Fabrizio Costa's EDeN.

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

import openbabel
import pybel
import json
import pybel
import networkx as nx
from networkx.readwrite import json_graph
import tempfile
import ioskgraph
import unicodedata

def obabel_to_eden(input, file_type = 'sdf',dict_labels={}, counter=[1], **options):
    """
    Takes a string list in sdf format and yields networkx graphs.
    Parameters
    ----------
    input : string
        A pointer to the data source.
    """
    
    f = ioskgraph.read(input)
    for line in f:
        if line.strip():
            l=unicodedata.normalize('NFKD', line).encode('ascii','ignore')
            mol=pybel.readstring(file_type,l)
            #print mol
            #remove hydrogens
            #mol.removeh()
            G = obabel_to_networkx(mol,dict_labels,counter)
            if len(G):
                yield G


def obabel_to_networkx( mol, dict_labels={},  counter=[1]):
    """
    Takes a pybel molecule object and converts it into a networkx graph.
    """
    #print "ObabelToNBetworkx, counter = ", counter 
    g = nx.Graph()
    g.graph['ordered']=False

    #atoms
    for atom in mol:
        #label = str(atom.type)
        label = str(atom.atomicnum)
        if label not in dict_labels:
            #print "new label", label, "key", counter
            dict_labels[label]= counter[0]
            counter[0]+=1
        g.add_node(atom.idx, label=label, viewpoint=True)
    #bonds
    edges = []
    bondorders = []
    for bond in openbabel.OBMolBondIter(mol.OBMol):
        label = str(bond.GetBO())
        g.add_edge( bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label = label )
    return g
