# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:40:07 2015
This file comes from Fabrizio Costa's EDeN.

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

from . import _ortho_NSPDKgraph as graph
"""
    Transforms labeled, weighted, nested graphs in sparse vectors.
"""
class NSPDKVectorizer():
    
    def __init__(self,
                     r = 3,
                     d = 3,
                     min_r = 0,
                     min_d = 0,
                     nbits = 20,
                     normalization = True,
                     inner_normalization = True,
                     pure_neighborhood_features = False,
                     discretization_size = 0,
                     discretization_dimension = 1):
                         self.vectObject=graph.OrthogonalVectorizer(
                                           r ,
                                           d ,
                                           min_r ,
                                           min_d ,
                                           nbits ,
                                           normalization ,
                                           inner_normalization ,
                                           pure_neighborhood_features ,
                                           discretization_size ,
                                           discretization_dimension)

    def transform(self, G_list, n_jobs = 1):
         return self.vectObject.transform(G_list,n_jobs)
         
    
"""
        Parameters
        ----------
        r : int 
            The maximal radius size.

        d : int 
            The maximal distance size.

        min_r : int 
            The minimal radius size.

        min_d : int 
            The minimal distance size.

        nbits : int 
            The number of bits that defines the feature space size: |feature space|=2^nbits.

        normalization : bool 
            If set the resulting feature vector will have unit euclidean norm.

        inner_normalization : bool 
            If set the feature vector for a specific combination of the radius and 
            distance size will have unit euclidean norm.
            When used together with the 'normalization' flag it will be applied first and 
            then the resulting feature vector will be normalized.

        pure_neighborhood_features : bool 
            If set additional features are going to be generated. 
            These features are generated in a similar fashion as the base features, 
            with the caveat that the first neighborhood is omitted.
            The purpose of these features is to allow vertices that have similar contexts to be 
            matched, even when they are completely different. 

        discretization_size : int
            Number of discretization levels for real vector labels.
            If 0 then treat all labels as strings. 

        discretization_dimension : int
            Size of the discretized label vector.
"""
