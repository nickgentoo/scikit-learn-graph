# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:02:41 2015

This file is from Fabrizio Costa's EDeN.

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
import requests
#import requests_cache
import numpy as np

#requests_cache.install_cache('bioinfogen_cache')

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
    
def dump_svmlight_file(kernel_matrix, target_array, file_path):
    """
    Saves a kernel matrix to file in the svmlight sparse format
    Parameters
    ----------
    kernel_matrix : numpy.array / numpy.matrix
        The N x N kernel matrix to be saved
    target_array : numpy.array
        An N-size array with the labels associated with each sample used to build the kernel matrix
    file_path : string
        The path where to save the matrix, should end with ".svmlight"
    """

    output = open(file_path, "w")

    for i in xrange(len(kernel_matrix)):
        output.write(str(target_array[i])+" ")

        for j in range(len(kernel_matrix[i])):
            if kernel_matrix[i][j] != 0.:
                output.write(str(j)+":"+str(kernel_matrix[i][j])+" ")

        output.write("\n")

    output.close()

#def store_matrix(matrix = '', output_dir_path = '', out_file_name = '', output_format = ''):
#    """
#    TODO: output of a matrix on a file.
#    """
#    return eden_io.store_matrix(matrix, output_dir_path, out_file_name, output_format)
#
#
