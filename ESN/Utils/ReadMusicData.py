import sys
sys.path.insert(0, '../')
import scipy
import scipy.sparse as sparse
import theano
from midi.utils import midiread, midiwrite
import glob
import cPickle
import numpy as np
dataset='../../../data/Nottingham/Nottingham.pickle'
trainingSet=glob.glob('../../../data/Nottingham/training_debug/*.mid')
testSet=glob.glob('../../../data/Nottingham/test_debug/*.mid')
validSet=glob.glob('../../../data/Nottingham/valid_debug/*.mid')


def loadDataSet(files):
    #File e il path della carterlla contente i file (*.mid)
    assert len(files) > 0, 'Training set is empty!' \
                           ' (did you download the data files?)'
    #mi calcolo quel'el 'esempio di lunghezza massima
    maxLen=0
    dataset=[]
    length=[]
    for f in files:
        currentMidi=midiread(f, (21, 109),0.3).piano_roll.astype(np.float64)
        dataset.append(currentMidi)
        length.append(currentMidi.shape[0])
        if maxLen<currentMidi.shape[0]:
            maxLen=currentMidi.shape[0]
    print "MAXLEN: ",maxLen
    return dataset,maxLen,len(dataset),length
  

def createTimeStep(nonZeroVal=[],timeStepLen=88):
  
  timeStep=np.zeros(88,)
  for element in nonZeroVal:
    #element is a value from 21 to 108
    timeStep[element-21]=1

  return timeStep

def createSeq(nonZeroValList=[]):
  seq=[]
  for timeStep in nonZeroValList:
    seq.append(createTimeStep(timeStep))
  return np.asarray(seq)
    
def pickleLoad(data):
  loadData=cPickle.load(file(data))
  trainSet=loadData['train']
  testSet=loadData['test']
  validSet=loadData['valid']
  trainSetOneZero=[]
  for seq in trainSet:
    trainSetOneZero.append(createSeq(seq))
  
  testSetOneZero=[]
  for seq in testSet:
    testSetOneZero.append(createSeq(seq))
  
  validSetOneZero=[]
  for seq in validSet:
    validSetOneZero.append(createSeq(seq))
  
  return trainSetOneZero,testSetOneZero,validSetOneZero
  
if __name__=="__main__":
    print pickleLoad(dataset)
  
    
    #dataset,maxLen,nSample,sampleLen=loadDataSet(trainingSet)
    #d=dict()
    #for seq,leng in zip(dataset,sampleLen):

      #d[leng]=seq
      
    #print sorted(d)# mi restituisce la lista deglle lunghezze, che qui sono le mie key, ordinate
    