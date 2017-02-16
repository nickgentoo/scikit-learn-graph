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


def dispatch(dataset):
    if dataset=="Pamitrain":
        print "Loading Pami(training set) dataset"
        g_it=load_trees_Pami_train()
    elif dataset == "Pamitrainincr":
            print "Loading Pami(training set) dataset"
            g_it = load_trees_Pami_train_incr()
    elif dataset=="Pamitest":
        print "Loading Pami(test set) dataset"
        g_it=load_trees_Pami_test()
    elif dataset == "Pamitestincr":
        print "Loading Pami(test set) dataset"
        g_it = load_trees_Pami_test_incr()
    elif dataset == "Pamivalid":
        print "Loading Pami(test set) dataset"
        g_it = load_trees_Pami_valid()
    elif dataset == "Pamivalidincr":
        print "Loading Pami(test set) dataset"
        g_it = load_trees_Pami_valid_incr()
    elif dataset == "Pamireduced":
        print "Loading Pami(test set) dataset"
        g_it = load_trees_Pami_testreduced()
    elif dataset == "Pamireducedincr":
        print "Loading Pami(test set) dataset"
        g_it = load_trees_Pami_testreduced_incr()

    else:
        print "Unknown dataset name"
    return g_it

def load_trees_Pami_train():
    """

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_data_url= 'http://www.math.unipd.it/~nnavarin/datasets/Pami/pami2010.4M.train.novec'
    #input_target_url='datasets/ENZYMES.labels'
    #input_data_url='datasets/ENZYMES.gspan'
    dat=dataset_tree(input_data_url,input_type='url')
    print 'Loaded Pami 4M tree dataset (NLP) for tree classification.'
    print len(dat.trees),'trees.'
    from sklearn.datasets.base import Bunch
    return Bunch(graphs=[i for i in dat.trees],
    target=dat.target)

def load_trees_Pami_train_incr():
    """

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_data_url= 'http://www.math.unipd.it/~nnavarin/datasets/Pami/pami2010.4M.train.novec'
    #input_target_url='datasets/ENZYMES.labels'
    #input_data_url='datasets/ENZYMES.gspan'
    dat=dataset_tree_incr(input_data_url,input_type='url')
    print 'Loaded Pami 4M tree dataset (NLP) for tree classification.'
    #print len(dat.trees),'trees.'
    from sklearn.datasets.base import Bunch
    return dat
def load_trees_Pami_test():
    """

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_data_url= 'http://www.math.unipd.it/~nnavarin/datasets/Pami/pami2010.test.novec'
    #input_target_url='datasets/ENZYMES.labels'
    #input_data_url='datasets/ENZYMES.gspan'
    dat=dataset_tree(input_data_url,input_type='url')
    print 'Loaded Pami test tree dataset (NLP) for tree classification.'
    print len(dat.trees),'trees.'
    from sklearn.datasets.base import Bunch
    return Bunch(graphs=[i for i in dat.trees],
    target=dat.target)
def load_trees_Pami_test_incr():
    """

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_data_url= 'http://www.math.unipd.it/~nnavarin/datasets/Pami/pami2010.test.novec'
    #input_target_url='datasets/ENZYMES.labels'
    #input_data_url='datasets/ENZYMES.gspan'
    dat=dataset_tree_incr(input_data_url,input_type='url')
    print 'Loaded Pami test tree dataset (NLP) for tree classification.'
    #print len(dat.trees),'trees.'
    from sklearn.datasets.base import Bunch
    return dat

def load_trees_Pami_testreduced():
    """

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_data_url= 'http://www.math.unipd.it/~nnavarin/datasets/Pami/pami2010.test.novec.reduced'
    #input_target_url='datasets/ENZYMES.labels'
    #input_data_url='datasets/ENZYMES.gspan'
    dat=dataset_tree(input_data_url,input_type='url')
    print 'Loaded Pami test reduced tree dataset (NLP) for tree classification.'
    print len(dat.trees),'trees.'
    from sklearn.datasets.base import Bunch
    return Bunch(graphs=[i for i in dat.trees],
    target=dat.target)
def load_trees_Pami_testreduced_incr():
    """

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_data_url= 'http://www.math.unipd.it/~nnavarin/datasets/Pami/pami2010.test.novec.reduced'
    #input_target_url='datasets/ENZYMES.labels'
    #input_data_url='datasets/ENZYMES.gspan'
    dat=dataset_tree_incr(input_data_url,input_type='url')
    print 'Loaded Pami test reduced tree dataset (NLP) for tree classification.'
    #print len(dat.trees),'trees.'
    from sklearn.datasets.base import Bunch
    return dat
def load_trees_Pami_valid():
    """

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_data_url= 'http://www.math.unipd.it/~nnavarin/datasets/Pami/pami2010.valid.novec'
    #input_target_url='datasets/ENZYMES.labels'
    #input_data_url='datasets/ENZYMES.gspan'
    dat=dataset_tree(input_data_url,input_type='url')
    print 'Loaded Pami validation tree dataset (NLP) for tree classification.'
    print len(dat.trees),'trees.'
    from sklearn.datasets.base import Bunch
    return Bunch(graphs=[i for i in dat.trees],
    target=dat.target)

def load_trees_Pami_valid_incr():
    """

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_data_url= 'http://www.math.unipd.it/~nnavarin/datasets/Pami/pami2010.valid.novec'
    #input_target_url='datasets/ENZYMES.labels'
    #input_data_url='datasets/ENZYMES.gspan'
    dat=dataset_tree_incr(input_data_url,input_type='url')
    print 'Loaded Pami validation tree dataset (NLP) for tree classification.'
    #print len(dat.trees),'trees.'
    from sklearn.datasets.base import Bunch
    return dat
def load_trees_CAvsCM():
    """

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/TREE/CAvsCM.prolog'
    #input_target_url='datasets/ENZYMES.labels'  
    #input_data_url='datasets/ENZYMES.gspan'
    dat=dataset_tree(input_data_url,input_type='url')
    print 'Loaded CAvsCM tree dataset for graph classification.'
    print len(dat.trees),'trees.'
    from sklearn.datasets.base import Bunch
    return Bunch(graphs=[i for i in dat.trees],
    target=dat.target)
def load_trees_inex2005_train():
    """

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/inex2005_train.svmlight.prolog'
    #input_target_url='datasets/ENZYMES.labels'  
    #input_data_url='datasets/ENZYMES.gspan'
    dat=dataset_tree(input_data_url,input_type='url')
    print 'Loaded INEX2005 train dataset for graph classification.'
    print len(dat.trees),'trees.'
    from sklearn.datasets.base import Bunch
    return Bunch(graphs=[i for i in dat.trees],
    target=dat.target)


class dataset_tree():
     def __init__(self,name,input_type='file'):
         input_types = ['url','file']
         assert(input_type in input_types),'ERROR: input_type must be one of %s ' % input_types
        
         if input_type is 'file':
                f = open(input,'r')
         elif input_type is 'url':
                import requests
                #f = requests.get(name).text.split('\n')
                f = requests.get(name).text.split('\n')

         #self.loadFromFile('datasets/inex2005.trainingset.500')
         self.loadFromFile(f)
         if input_type is 'file':
                f.close()


    
    
     def loadFromFile(self, f):
            """
            Load a set of strings from a file and creates a list of examples objects. File format is specified by the method loadExample().
             : Parameter
            filename: the name of the file containing the set of strings in prolog format
            """
            #self.filename = filenamepython -m skgraph/Unity_tests/load_tree_dataset
            self.trees = []
            self.target = []
            #f = open(filename,"r")
            for line in f:
                if line.strip():
                    ex,target=self.TreeFromPrologString(line)
                    self.trees.append(ex)
                    self.target.append(target)

    
            
     def loadExample(self, line):
            return self.TreeFromPrologString(line)
     def TreeFromPrologString(self, s):
            """
            Create a Tree object from a string representing a tree in prolog format (see below for a description of the format)
             : Parameter
            s: the string encoding the subtree and its label. The format of the string is the following
               1) [+-0-9]: numerical target label
               2) " "|\t 
               3) subtree in prolog format (see TreeNodeFromPrologString class for a description).
            """
            target, i, tmps = ("", 0, "")
            while s[i] in ".-+0123456789": #looking for numeric target value
                tmps += s[i]
                i += 1
            if len(tmps) > 0 and (s[i] == " " or s[i] == "\t"): #the target is valid
                target = tmps
                i+=1
            else:
                i=0
            tree = self.TreeNodeFromPrologString(s[i:])
            target = float(target)
            return (tree,target)
            
     def TreeNodeFromPrologString(self, s, symboltable="(),"):
        """
        A class for creating TreeNode objects starting from strings in prolog format.
        The prolog format is as follows: 
        1) [a-zA-Z0-9]+: the label of the node
        2) (: beginning of a subtree (if there is any)
        3) ,: beginning of a sibling node (if there is any)
        4) ): end of a subtree (if there is any)
        Note that the chars (), can be changed by the user.
        """
        """
        This method creates a set of TreeNode objects corresponding to a subtree as described by the string s.
         : Parameter
        s: the string encoding the subtree
        symboltable (optional): a 3-char string where the first char corresponds to the beginning of a new subtree, 
                                the second corresponds to the end of a subtree and the third separates two siblings
        """
        from networkx import DiGraph
        tree=DiGraph()
        startSubtreeChar,endSubtreeChar,startSiblingChar=symboltable
        s = s.rstrip('\n') #remove trailing newlines
        i, lens = (0, len(s))
        node_id=1
        aa = []
        while (i < lens): 
            tmps = ""
            while (i < lens) and s[i] not in symboltable:
                tmps += s[i]
                i += 1
            if len(tmps) > 0:
                #print 'label', tmps
                tree.add_node(node_id,label=tmps,childrenOrder=[]) #t = TreeNode(tmps,[])
                if node_id==1:
                    tree.graph['root']=node_id
                node_id+=1                
                if len(aa)>0:
                    tree.add_edge(aa[-1],node_id-1) #-1 is the last item
                    tree.node[aa[-1]]['childrenOrder'].append(node_id-1)

            if i < lens:
                if s[i] == startSubtreeChar: 
                    aa.append(node_id-1)
                elif s[i] == endSubtreeChar:
                    t=aa.pop()
                elif s[i] == startSiblingChar: 
                    pass
            i += 1
        return tree

class dataset_tree_incr():
         def __init__(self, name, input_type='file'):
             self.name=name
             input_types = ['url', 'file']
             assert (input_type in input_types), 'ERROR: input_type must be one of %s ' % input_types
             self.input_type=input_type

         def nextelement(self):
             """
             Load a set of strings from a file and creates a list of examples objects. File format is specified by the method loadExample().
              : Parameter
             filename: the name of the file containing the set of strings in prolog format
             """
             # self.filename = filenamepython -m skgraph/Unity_tests/load_tree_dataset
             if self.input_type is 'file':
                 f = open(input, 'r')
             elif self.input_type is 'url':
                 import requests
                 # f = requests.get(name).text.split('\n')
                 f = requests.get(self.name).text.split('\n')
             # f = open(filename,"r")
             for line in f:
                 if line.strip():
                     ex, target = self.TreeFromPrologString(line)
                     yield(ex,target)

             if self.input_type is 'file':
                f.close()

         def loadExample(self, line):
             return self.TreeFromPrologString(line)

         def TreeFromPrologString(self, s):
             """
             Create a Tree object from a string representing a tree in prolog format (see below for a description of the format)
              : Parameter
             s: the string encoding the subtree and its label. The format of the string is the following
                1) [+-0-9]: numerical target label
                2) " "|\t
                3) subtree in prolog format (see TreeNodeFromPrologString class for a description).
             """
             target, i, tmps = ("", 0, "")
             while s[i] in ".-+0123456789":  # looking for numeric target value
                 tmps += s[i]
                 i += 1
             if len(tmps) > 0 and (s[i] == " " or s[i] == "\t"):  # the target is valid
                 target = tmps
                 i += 1
             else:
                 i = 0
             tree = self.TreeNodeFromPrologString(s[i:])
             target = float(target)
             return (tree, target)

         def TreeNodeFromPrologString(self, s, symboltable="(),"):
             """
             A class for creating TreeNode objects starting from strings in prolog format.
             The prolog format is as follows:
             1) [a-zA-Z0-9]+: the label of the node
             2) (: beginning of a subtree (if there is any)
             3) ,: beginning of a sibling node (if there is any)
             4) ): end of a subtree (if there is any)
             Note that the chars (), can be changed by the user.
             """
             """
             This method creates a set of TreeNode objects corresponding to a subtree as described by the string s.
              : Parameter
             s: the string encoding the subtree
             symboltable (optional): a 3-char string where the first char corresponds to the beginning of a new subtree,
                                     the second corresponds to the end of a subtree and the third separates two siblings
             """
             from networkx import DiGraph
             tree = DiGraph()
             startSubtreeChar, endSubtreeChar, startSiblingChar = symboltable
             s = s.rstrip('\n')  # remove trailing newlines
             i, lens = (0, len(s))
             node_id = 1
             aa = []
             while (i < lens):
                 tmps = ""
                 while (i < lens) and s[i] not in symboltable:
                     tmps += s[i]
                     i += 1
                 if len(tmps) > 0:
                     # print 'label', tmps
                     tree.add_node(node_id, label=tmps, childrenOrder=[])  # t = TreeNode(tmps,[])
                     if node_id == 1:
                         tree.graph['root'] = node_id
                     node_id += 1
                     if len(aa) > 0:
                         tree.add_edge(aa[-1], node_id - 1)  # -1 is the last item
                         tree.node[aa[-1]]['childrenOrder'].append(node_id - 1)

                 if i < lens:
                     if s[i] == startSubtreeChar:
                         aa.append(node_id - 1)
                     elif s[i] == endSubtreeChar:
                         t = aa.pop()
                     elif s[i] == startSiblingChar:
                         pass
                 i += 1
             return tree