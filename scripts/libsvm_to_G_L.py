# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:50:38 2015

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
import sys
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
if len(sys.argv)<2:
    sys.exit("python cross_validation_from_matrix_norm.py inputMatrix.libsvm")

##TODO read from libsvm format
fname=sys.argv[1]
from sklearn.datasets import load_svmlight_file
print "Reading libsvm file.."
_km, target_array = load_svmlight_file(sys.argv[1])
km=_km[:,1:].todense()

#print type(km)
print "saving matrices.."
np.savetxt(fname+".G",km,fmt="%g")
np.savetxt(fname+".L",target_array,fmt="%g")
