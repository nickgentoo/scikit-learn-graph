# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 15:42:13 2014

@author: Nicolò Navarin
    This file is an encapsulatorr. It defines the functions for the input/output of labels
    (arrays) and matrices
    In the current version it uses pyEDeN functions.  
"""
import requests
import numpy as np
def read(uri):
    """
    Abstract read function. EDeN can accept a URL, a file path and a python list.
    In all cases an iteratatable object should be returned.
    """
    if hasattr(uri, '__iter__'):
        # test if it is iterable: works for lists and generators, but not for
        # strings
        return uri
    else:
        try:
            # try if it is a URL and if we can open it
            f = requests.get(uri).text.split('\n')
        except ValueError:
            # assume it is a file object
            f = open(uri)
        return f
def load_target(name):
    """
    Return a numpy array of integers to be used as target vector.
    Parameters
    ----------
    name : string
        A pointer to the data source.
    """

    Y = [y.strip() for y in read(name) if y]
    return np.array(Y).astype(int)
    
#def store_matrix(matrix = '', output_dir_path = '', out_file_name = '', output_format = ''):
#    """
#    TODO: output of a matrix on a file.
#    """
#    return eden_io.store_matrix(matrix, output_dir_path, out_file_name, output_format)
#
#
