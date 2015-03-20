# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 17:13:51 2015

@author: nick
"""
class dataset_tree_inex2005():
     def __init__(self):
         self.loadFromFile('inex2005.trainingset.500')
    
    
     def loadFromFile(self, filename):
            """
            Load a set of strings from a file and creates a list of examples objects. File format is specified by the method loadExample().
             : Parameter
            filename: the name of the file containing the set of strings in prolog format
            """
            self.filename = filename
            self.trees = []
            f = open(filename,"r")
            for line in f:
                ex,target=self.TreeFromPrologString(line)
                self.trees.append(ex)
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