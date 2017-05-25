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
import sys
from skgraph.feature_extraction.graph.ODDSTVectorizer import ODDSTVectorizer
from skgraph.feature_extraction.graph.NSPDK.NSPDKVectorizer import NSPDKVectorizer
from skgraph.feature_extraction.graph.WLVectorizer import WLVectorizer

from skgraph.datasets import load_graph_datasets
import numpy as np
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers import Input
from keras.models import Model
from keras import regularizers

#tutorial from https://blog.keras.io/building-autoencoders-in-keras.html

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
        sys.exit("python ODDKernel_example.py dataset r l kernel")
    dataset=sys.argv[1]
    max_radius=int(sys.argv[2])
    la=float(sys.argv[3])
    #hashs=int(sys.argv[3])
    njobs=1
    name=str(sys.argv[4])

    kernel=sys.argv[5]
    #FIXED PARAMETERS
    normalization=True
    g_it=load_graph_datasets.dispatch(dataset)



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
       
    features=Vectorizer.transform(g_it.graphs) #Parallel ,njobs
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
    print "Extracted", features.shape[1], "features from",features.shape[0],"examples."
    n=features.shape[0]
    densefeat=features.todense()
    x_train=densefeat[:int(n*0.8),:]
    #TODO sbagliato, fare slicing!
    x_train = x_train.reshape((len(x_train), np.prod(features.shape[1])))
    x_test=densefeat[int(n*0.8):,:]

    print x_train.shape
    print x_test.shape
    AutoEncoder=DNN(x_train.shape[1],layerSize=[1000])

    AutoEncoder.model.fit(x_train, x_train,
                    nb_epoch=10,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    # encode and decode some digits
    # note that we take them from the *test* set

    encoded_features = AutoEncoder.encoder.predict(densefeat)
    print "Extracted", encoded_features.shape[1], "features from",encoded_features.shape[0],"examples."
    print "Saving Features in svmlight format in", name+".svmlight"
    #print GMsvm
    from sklearn import datasets
    datasets.dump_svmlight_file(encoded_features,g_it.target, name+".svmlight", zero_based=False)
