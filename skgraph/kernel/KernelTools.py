__author__ = "Riccardo Tesselli"
__date__ = "04/feb/2015"
__credits__ = ["Riccardo Tesselli"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer = "Riccardo Tesselli"
__email__ = "riccardo.tesselli@gmail.com"
__status__ = "Production"

import numpy as np
from operator import itemgetter
from scipy.sparse import csr_matrix, coo_matrix

def computeFeaturesWeights(svindexes,coeflist,dictfeatures):
    """
    Function that computes the relevance score w_j=sum_over_support_graphs_of(alpha*y*phi(G)_j) 
    """
    features=[c for (r,c) in dictfeatures.keys()]
    features=np.unique(features)
    weights={}
    for f in features:
        weight=0
        coefindex=0
        for svi in svindexes:
            if not dictfeatures.get((svi,f)) is None:
                weight+=coeflist[coefindex]*dictfeatures.get((svi,f))
            coefindex+=1
        weights[f]=weight
    return weights

def topWeights(number,weights,positive=True):
    listweights=weights.items()
    listweights.sort(key=itemgetter(1))
    if positive:
        listweights=listweights[-number:]
    else:
        listweights=listweights[:number]
    return dict(listweights)
    
def convert_to_sparse_matrix_enc(feature_dict, MapEncToId=None):
        """
        Function that convert the feature vector from dictionary to sparse matrix
        @type feature_dict: Dictionary
        @param feature_dict: a feature vector
        
        @type MapEncToId: self.UniqueMap
        @param MapEncToId: Map between feature's encodings and integer values
        
        @rtype: scipy.sparse.csr_matrix
        @return: the feature vector in sparse form
        """
        if len(feature_dict) == 0:
            raise Exception('ERROR: something went wrong, empty feature_dict.')
        data = feature_dict.values()
        row, col = [], []
        if not MapEncToId is None:
            for i, j in feature_dict.iterkeys():
                row.append( i )
                col.append( MapEncToId[j] )
            X = csr_matrix( (data,(row,col)), shape = (max(row)+1, max(col)+1))
            return X, MapEncToId
        else:
            for i, j in feature_dict.iterkeys():
                row.append( i )
                col.append( j )
            MapEncToId={}
            idenc=0
            for enc in np.unique(col):
                MapEncToId[enc]=idenc
                idenc+=1
            colid=[]
            for enc in col:
                colid.append(MapEncToId[enc])
            X = csr_matrix( (data,(row,colid)), shape = (max(row)+1, max(colid)+1))
            return X, MapEncToId
            
def convert_to_sparse_matrix(feature_dict, MapEncToId=None):
        """
        Function that convert the feature vector from dictionary to sparse matrix
        @type feature_dict: Dictionary
        @param feature_dict: a feature vector
        
        @type MapEncToId: self.UniqueMap
        @param MapEncToId: Map between feature's encodings and integer values
        
        @rtype: scipy.sparse.csr_matrix
        @return: the feature vector in sparse form
        """
        if len(feature_dict) == 0:
            raise Exception('ERROR: something went wrong, empty feature_dict.')
        data = feature_dict.values()
        row, col = [], []
        if not MapEncToId is None:
            for i, j in feature_dict.iterkeys():
                row.append( i )
                col.append( MapEncToId[j]) 
            X = csr_matrix( (data,(row,col)), shape = (max(row)+1, max(col)+1))
            return X
        else:
            for i, j in feature_dict.iterkeys():
                row.append( i )
                col.append( j )
            MapEncToId={}
            idenc=0
            for enc in np.unique(col):
                MapEncToId[enc]=idenc
                idenc+=1
            colid=[]
            for enc in col:
                colid.append(MapEncToId[enc])
            X = csr_matrix( (data,(row,colid)), shape = (max(row)+1, max(colid)+1))
            return X
def convert_to_sparse_matrix_list(feature_dict, MapEncToId=None):
        """
        Function that convert the feature vector from dictionary to sparse matrix
        @type feature_dict: Dictionary
        @param feature_dict: a feature vector
        
        @type MapEncToId: self.UniqueMap
        @param MapEncToId: Map between feature's encodings and integer values
        
        @rtype: scipy.sparse.csr_matrix
        @return: the feature vector in sparse form
        """
        if len(feature_dict) == 0:
            raise Exception('ERROR: something went wrong, empty feature_dict.')
        data = feature_dict.values()
        row, col = [], []
        if not MapEncToId is None:
            for i, j in feature_dict.iterkeys():
                row.append( i )
                col.append( MapEncToId[j]) 
            X = csr_matrix( (data,(row,col)), shape = (max(row)+1, max(col)+1), dtype=np.dtype(object))
            return X
        else:
            for i, j in feature_dict.iterkeys():
                row.append( i )
                col.append( j )
            MapEncToId={}
            idenc=0
            for enc in np.unique(col):
                MapEncToId[enc]=idenc
                idenc+=1
            colid=[]
            for enc in col:
                colid.append(MapEncToId[enc])
            X = csr_matrix( (data,(row,colid)), shape = (max(row)+1, max(colid)+1), dtype=np.dtype(object))
            return X