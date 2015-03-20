"""
This file comes from Fabrizio Costa's pyEDeN.

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

from ..datasets import ioskgraph
import json
import networkx as nx
#from eden import util

def gspan_to_eden(infile):
    """
    Takes a string list in the extended gSpan format and yields networkx graphs.

    Parameters
    ----------
    input : string
        A pointer to the data source.

    """

    string_list = []
    for line in ioskgraph.read(infile):
        if line.strip():
            if line[0] in ['g', 't']:
                if string_list:
                    yield _gspan_to_networkx(string_list)
                string_list = []
            string_list += [line]

    if string_list:
        yield _gspan_to_networkx(string_list)


def _gspan_to_networkx(string_list):
    """
    Utility function that generates a single networkx graph from a string
    Parameters
    ----------
    string_list : list of string
        A string list encoding a gspan graph.
    """
    graph = nx.Graph()
    graph.graph['ordered']=False

    for line in string_list:
        if line.strip():
            line_list = line.split()
            firstcharacter = line_list[0]
            #process vertices
            if firstcharacter in ['v', 'V']:
                vid = int(line_list[1])
                vlabel = line_list[2]
                #lowercase v indicates active viewpoint
                if firstcharacter == 'v':
                    weight = 1
                else: #uppercase v indicates no-viewpoint
                    weight = 0.1
                graph.add_node(vid, label=vlabel, weight=weight, viewpoint=True)
                #abstract vertices
                if vlabel[0] == '^':
                    graph.node[vid]['nesting'] = True
                #extract the rest of the line  as a JSON
                #string that contains all attributes
                attribute_str = ' '.join(line_list[3:])
                if attribute_str.strip():
                    attribute_dict = json.loads(attribute_str)
                    graph.node[vid].update(attribute_dict)
            #process edges
            if firstcharacter == 'e':
                srcid = int(line_list[1])
                destid = int(line_list[2])
                elabel = line_list[3]
                graph.add_edge(srcid, destid, label=elabel)
                attribute_str = ' '.join(line_list[4:])
                if attribute_str.strip():
                    attribute_dict = json.loads(attribute_str)
                    graph.edge[srcid][destid].update(attribute_dict)
    assert(len(graph) > 0), 'ERROR: generated empty graph. Perhaps wrong format?'
    return graph
