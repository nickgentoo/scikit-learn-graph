import sys
sys.path.insert(0, '../')
import numpy as np
np.set_printoptions(threshold=np.nan)
import scipy
import scipy.sparse as sparse


def stepRound(x):
  if x<0.5:
    return 0
  else:
    return 1 
def accEval(targets,predictions,ftr):
   Acc=[]
   for tar,pred in zip(targets,predictions):
      #for i in range(10):
      #print "seq:",seq
      #print "tar:",tar
      #raw_input("-----")
      value=evaluateAcc(tar,pred)
      #print value
      #print len(tar)
      Acc.append(value)
      #print >>f, ('esempio:', i, 'Acc:', value)
      print>>ftr,"Acc:"+str(value)+", len:"+str(len(tar))
   print>>ftr,"Avg: ",np.mean(Acc)
   return value

def accEvalSingleSample(groundTruth,data,ftr):
   value=evaluateAcc(groundTruth, data)
   print>>ftr,(value)
   return value


def evaluateAcc (groundTruth=None, generate=None ):
    #Poniamo vero che i parametri in input siano dei piano-roll, quindi abbiamo un vettore che contiene piu vettori di 88 valori (o cmq io considero solo i primi 88 (che sono r[0]-r[1])

    assert  len(groundTruth) == len(generate), 'La dimensione dei due input non corrisposnde'
    #maxPoli=0
    #clipPoly=[]
    #GtPoly=[]
    TP=0.0
    FP=0.0
    FN=0.0
    
    
    #print "pred:",generate
    #print "tar:",groundTruth
    
    #for frame, gt in zip(groundTruth, generate):
    for gt,pred in zip(groundTruth, generate):   
        #calcolo massima polifonia della clip
        #currentPoliGT=0
        #currentPoliGen=0
        for i in xrange(len(gt)):
	    normPred=stepRound(pred[i])
            #if frame[i]==1:
            if normPred==1:
                #currentPoliGen=currentPoliGen+1
                if gt[i]==1:
                    TP=TP+1
                    #currentPoliGT=currentPoliGT+1
                if gt[i]==0:
                    FP=FP+1
            else:
                if gt[i]==1:
		    FN=FN+1
                    #currentPoliGT=currentPoliGT+1
#                    else:
#                        #vuol dire che sia frame[i] che gt[i] sono uguali a 0, quindi devo aumentare i TP
#                        TP=TP+1
        #clipPoly.append(currentPoliGen)
        #GtPoly.append(currentPoliGT)
        #if currentPoliGT>maxPoli:
            #maxPoli=currentPoliGT
    
    #for i, frame in enumerate(groundTruth):
        #FN=FN+(maxPoli-clipPoly[i])-(maxPoli-GtPoly[i])

    #print 'TP=', TP
    #print 'FP=', FP
    
    #print 'FN=', FN

    ACC=0.0
    if TP+FP+FN==0:
      #FP=0.0001
      #e' l'unico caso in cui tutto il gt e' a 0 e tutta la predizione e' tutta a 0
      ACC=1.0
    else:
      ACC=TP/(TP+FP+FN)
    #ACC=TP/(TP+FP+FN)
    #print ACC
    #raw_input("-------------")
    return ACC
  

  
if __name__=="__main__":
  
  tar=[np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]])]
  gen=[np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]])]
  dataset=[tar,gen]

  f=open("test",'w+')
  accEval(tar,gen,f)