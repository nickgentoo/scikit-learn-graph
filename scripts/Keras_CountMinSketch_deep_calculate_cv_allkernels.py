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
from copy import deepcopy
import sys
from skgraph.feature_extraction.graph.ODDSTVectorizer import ODDSTVectorizer
from skgraph.feature_extraction.graph.NSPDK.NSPDKVectorizer import NSPDKVectorizer
from skgraph.feature_extraction.graph.WLVectorizer import WLVectorizer
from keras import backend as K
from skgraph.datasets import load_graph_datasets
import numpy as np
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from countminsketch_Numpy import CountMinSketch
from keras.models import save_model, load_model

#tutorial from https://blog.keras.io/building-autoencoders-in-keras.html
class DeepNNVariableLayers:
  def __init__(self,inputDim,layerSize=None):
      # this is our input placeholder
    print layerSize
    self.encoder = Sequential()
    #input layer
    input_img = Input(shape=(inputDim,))
    encoded=input_img

    #encoded=Dense(layerSize[0], init='uniform', activation='relu')(input_img)
    #middle layers
    for size in layerSize[0:]: 
      encoded=Dropout(0.7)(encoded)

      encoded=Dense(size, activation='relu')(encoded)
    #output layer
    #decoded=Dense(layerSize[-2], init='uniform', activation='relu')(encoded)
    encoded1=Dropout(0.7)(encoded)
    encoded1= Dense(1, activation='sigmoid')(encoded1)

    first=True
    for size in reversed(layerSize[:-1]):  
      if (first==True):
          decoded=Dropout(0.7)(encoded)
          decoded=Dense(size,  activation='relu')(decoded) 

          first=False
      else:
          decoded=Dropout(0.7)(decoded)

          decoded=Dense(size, activation='relu')(decoded) 
    if (first==True):
        decoded=encoded
    decoded=Dropout(0.7)(decoded)
    decoded = Dense(inputDim, activation='relu')(decoded)

    self.model = Model(input=input_img, output=decoded)
    print self.model.summary()
    # this model maps an input to its encoded representation
    self.encoder = Model(input=input_img, output=encoded1)  
    print self.encoder.summary()
    self.model.compile(optimizer='adagrad', loss='binary_crossentropy') # adagrad binary_crossentropy
    self.encoder.compile(optimizer='adagrad', loss='binary_crossentropy',metrics=['accuracy']) # adagrad

    #self.model = Sequential()
    #self.model.add(Dense(layerSize[0], input_dim=inputDim, init='uniform', activation='tanh'))
    #for size in layerSize[1:]:  
    #  self.model.add(Dense(size, init='uniform', activation='tanh'))
#==============================================================================
# class DeepNN:
#   def __init__(self,inputDim,layerSize=None):
#       # this is our input placeholder
#     print layerSize
#     input_img = Input(shape=(inputDim,))
#     # "encoded" is the encoded representation of the input
#     encoded = Dense(layerSize[0], activation='relu')(input_img)
#     encoded = Dense(layerSize[1], activation='relu')(encoded)
#     encoded = Dense(layerSize[2], activation='relu')(encoded)
#     # add a Dense layer with a L1 activity regularizer
#     #encoded = Dense(layerSize[0], activation='relu',
#     #            activity_regularizer=regularizers.activity_l1(10e-3))(input_img)    
#     
#     # "decoded" is the lossy reconstruction of the input
#     decoded = Dense(layerSize[1], activation='relu')(encoded)
#     decoded = Dense(layerSize[0], activation='relu')(decoded)
#     decoded = Dense(inputDim, activation='sigmoid')(decoded)
#     # this model maps an input to its reconstruction
#     self.model = Model(input=input_img, output=decoded)
#     print self.model.summary()
#     # this model maps an input to its encoded representation
#     self.encoder = Model(input=input_img, output=encoded1)    
#     # create a placeholder for an encoded (32-dimensional) input
#     encoded_input = Input(shape=(layerSize[2],))
#     # retrieve the last layer of the autoencoder model
#     decoder_layer = self.model.layers[-3]
#     # create the decoder model
#     self.decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
#     #self.model.compile(optimizer='adadelta', loss='binary_crossentropy')
#     self.model.compile(optimizer='adagrad', loss='binary_crossentropy')
# 
#     #self.model = Sequential()
#     #self.model.add(Dense(layerSize[0], input_dim=inputDim, init='uniform', activation='tanh'))
#     #for size in layerSize[1:]:  
#     #  self.model.add(Dense(size, init='uniform', activation='tanh'))
# 
#       
#     #self.model.add(Dense(outputDim, init='uniform', activation='relu'))
#==============================================================================
class DNN:
  def __init__(self,inputDim,layerSize=None):
      # this is our input placeholder
    print layerSize
    input_img = Input(shape=(inputDim,))
    # "encoded" is the encoded representation of the input
    
    encoded = Dense(layerSize[0], activation='relu')(input_img)
    # add a Dense layer with a L1 activity regularizer
    #encoded = Dense(layerSize[0], activation='relu',
    #            activity_regularizer=regularizers.activity_l1(10e-1))(input_img)    
    
    # "decoded" is the lossy reconstruction of the input
    
    decoded = Dense(inputDim, activation='sigmoid')(encoded)
    # this model maps an input to its reconstruction
    self.model = Model(input=input_img, output=decoded)
    
    # this model maps an input to its encoded representation
    self.encoder = Model(input=input_img, output=encoded)    
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(layerSize[0],))
    # retrieve the last layer of the autoencoder model
    decoder_layer = self.model.layers[-1]
    # create the decoder model
    self.decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    #self.model.compile(optimizer='adadelta', loss='binary_crossentropy')
    self.model.compile(optimizer='adagrad', loss='binary_crossentropy')

    #self.model = Sequential()
    #self.model.add(Dense(layerSize[0], input_dim=inputDim, init='uniform', activation='tanh'))
    #for size in layerSize[1:]:  
    #  self.model.add(Dense(size, init='uniform', activation='tanh'))

      
    #self.model.add(Dense(outputDim, init='uniform', activation='relu'))

if __name__=='__main__':
    if len(sys.argv)<1:
        sys.exit("python ODDKernel_example.py dataset r l kernel list_n_hidden_as_string")
    dataset=sys.argv[1]
    max_radius=int(sys.argv[2])
    la=float(sys.argv[3])
    #hashs=int(sys.argv[3])
    njobs=1
    #name=str(sys.argv[4])

    kernel=sys.argv[4]
    n_hidden = map(int,sys.argv[5].split() )

    #n_hidden=int(sys.argv[6])

    #FIXED PARAMETERS
    normalization=True
    
    if dataset=="CAS":
        print "Loading bursi(CAS) dataset"        
        g_it=load_graph_datasets.load_graphs_bursi()
    elif dataset=="GDD":
        print "Loading GDD dataset"        
        g_it=load_graph_datasets.load_graphs_GDD()
    elif dataset=="CPDB":
        print "Loading CPDB dataset"        
        g_it=load_graph_datasets.load_graphs_CPDB()
    elif dataset=="AIDS":
        print "Loading AIDS dataset"        
        g_it=load_graph_datasets.load_graphs_AIDS()
    elif dataset=="NCI1":
        print "Loading NCI1 dataset"        
        g_it=load_graph_datasets.load_graphs_NCI1()
    elif dataset=="NCI109":
        print "Loading NCI109 dataset"        
        g_it=load_graph_datasets.load_graphs_NCI109()
    elif dataset=="NCI123":
        print "Loading NCI123 dataset"        
        g_it=load_graph_datasets.load_graphs_NCI123()
    elif dataset=="NCI_AIDS":
        print "Loading NCI_AIDS dataset"        
        g_it=load_graph_datasets.load_graphs_NCI_AIDS()
    else:
        print "Unknown dataset name"
     

    if kernel=="WL":
        print "Lambda ignored"
        print "Using WL fast subtree kernel"
        Vectorizer=WLVectorizer(r=max_radius,normalization=normalization)
    elif kernel=="ODDST":
        print "Using ST kernel"
        Vectorizer=ODDSTVectorizer(r=max_radius,l=la,normalization=normalization)
    elif kernel=="NSPDK":
        print "Using NSPDK kernel, lambda parameter interpreted as d"
        Vectorizer=NSPDKVectorizer(r=max_radius,d=int(la),normalization=normalization)
    else:
        print "Unrecognized kernel"
       
    m=4000
    d=1
    features=Vectorizer.transform(g_it.graphs) #Parallel ,njobs
    featnew=[]
    print "examples, features", features.shape

    for i in xrange(features.shape[0]):
        exCMS=CountMinSketch(m,d)

        ex=features[i][0]
        #W=csr_matrix(ex)
    
        rows,cols = ex.nonzero()
        dot=0.0
        for row,col in zip(rows,cols):
              #((row,col), ex[row,col])
              #print col, ex[row,col]
              #dot+=WCMS[col]*ex[row,col]
              exCMS.add(col,ex[row,col])
        featnew.append(exCMS.asarray())
    #print GM
#    GMsvm=[]    
#    for i in xrange(len(GM)):
#        GMsvm.append([])
#        GMsvm[i]=[i+1]
#        GMsvm[i].extend(GM[i])
#    #print GMsvm
#    from sklearn import datasets
#    print "Saving Gram matrix"
#    #datasets.dump_svmlight_file(GMsvm,g_it.target, name+".svmlight")
#    datasets.dump_svmlight_file(np.array(GMsvm),g_it.target, name+".svmlight")
#    #Test manual dump
    #LEARN AUTOENCODER
    #print featnew
    featnew=np.asarray(featnew)
    print "Extracted", features.shape[1], "features from",features.shape[0],"examples."
    n=featnew.shape[0]
    print "n examples", n
    densefeat=featnew #.todense()
    x_train=densefeat[:int(n*0.8),:]
    y_train=g_it.target[:int(n*0.8)]
    #TODO sbagliato, fare slicing!
    x_train = x_train.reshape((len(x_train), np.prod(featnew.shape[1])))
    x_test=densefeat[int(n*0.8):,:]
    y_test=g_it.target[int(n*0.8):]
    print x_train.shape
    print x_test.shape
    #AutoEncoder=DeepNN(x_train.shape[1],layerSize=[n_hidden[0],n_hidden[1],n_hidden[2]])
    AutoEncoder=DeepNNVariableLayers(x_train.shape[1],layerSize=n_hidden)

    AutoEncoder.model.fit(x_train, x_train,
                    nb_epoch=20,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    # encode and decode some digits
    # note that we take them from the *test* set
#    AutoEncoder.encoder.fit(x_train, y_train,
#                    nb_epoch=10,
#                    batch_size=128,
#                    shuffle=True,
#                    validation_data=(x_test, y_test))
    AutoEncoder.encoder.save("mymodel.h5")
    del AutoEncoder.model

    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    encoder = LabelEncoder()
    encoder.fit(g_it.target)
    encoded_Y = encoder.transform(g_it.target)
    def build():
        #return deepcopy(AutoEncoder.encoder)
        return load_model("mymodel.h5")
    # evaluate baseline model with standardized dataset
    np.random.seed(42)
    estimators = []
    #estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp',KerasClassifier(build_fn=build, nb_epoch=20, batch_size=256, verbose=1)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = cross_val_score(pipeline, densefeat, encoded_Y, cv=kfold)
    print
    print ("Mean, std: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
    
    
    
#    
#    
#    
#    
#    encoded_features = AutoEncoder.encoder.predict(densefeat)
#    print "features encoded in", encoded_features.shape[1], "features"
#    from sklearn import cross_validation
#    from sklearn.svm import SVC, LinearSVC
#    clf = LinearSVC(C=100,dual=True) #, class_weight='auto'
#    #clf = SVC(C=1,kernel='rbf',gamma=0.001) #, class_weight='auto'
##
#    y_train=g_it.target
##    kf = cross_validation.StratifiedKFold(y_train, n_folds=10, shuffle=True,random_state=42)
##    scores=cross_validation.cross_val_score(
##        clf, encoded_features, y_train, cv=kf, scoring='accuracy')
##    print scores
##    print "Inner AUROC: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())
#
#    #print GM
###############################################################################
## Train classifiers
##
## For an initial search, a logarithmic grid with basis
## 10 is often helpful. Using a basis of 2, a finer
## tuning can be achieved but at a much higher cost.
#    from sklearn.cross_validation import StratifiedShuffleSplit
#    from sklearn.grid_search import GridSearchCV
#    C_range = np.logspace(-2, 4, 7)
##    gamma_range = np.logspace(-9, 3, 13)
#    param_grid = dict( C=C_range)
#    cv = StratifiedShuffleSplit(y_train, n_iter=10, test_size=0.2, random_state=42)
#    grid = GridSearchCV(LinearSVC(), param_grid=param_grid, cv=cv,verbose=10)
#    print "starting grid search"
#    grid.fit(encoded_features, y_train)
##    
#    print("The best parameters are %s with a score of %0.4f"
#          % (grid.best_params_, grid.best_score_))