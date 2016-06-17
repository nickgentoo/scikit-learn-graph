import theano
import theano.tensor as T
import numpy as np
import numpy.linalg as linalg
import scipy.sparse 
import math
 #TODO: creare le matrici dei pesi iniziali che rispettino la  ESP
outFun= lambda x: (1/(1+T.exp(-x*0.1)))*10-(10/2)
class EchoStateNetwork:
  def __init__ (self,input_dim,resevoir_dim,output_dim,activation_function=T.nnet.sigmoid,activation_output=lambda x:x,scaleIn=1,scaleRes=1):
    self.input_dim=input_dim
    self.resevoir_dim=resevoir_dim
    self.output_dim=output_dim
    self.activation_function=activation_function
    self.activation_output=activation_output

    self.W=theano.shared(self.ECPMatrix(np.random.uniform(low=-scaleRes, high=scaleRes,size=(resevoir_dim,resevoir_dim))))
    self.W_in=theano.shared(self.ECPMatrix(np.random.uniform(low=-scaleIn, high=scaleIn,size=(input_dim,resevoir_dim))))
    self.W_fb=theano.shared(self.ECPMatrix(np.random.uniform(low=-scaleRes, high=scaleRes,size=(output_dim,resevoir_dim))))
    self.W_out=theano.shared(np.random.uniform(low=-1/2, high=1/2,size=(resevoir_dim,output_dim)))
    self.__theano_build__()
  
  def ECPMatrix(self,m,epsilon=0.03):
    #si calcola svd, si fa si che tutti gli autovalori siano inferriori a 1 e si richrea la matrice
    v,s,u_t=linalg.svd(m)
    norm=s[0]+epsilon
    newS=np.zeros((v.shape[1],u_t.shape[0]))
    for i,element in enumerate(s):
      newS[i,i]=element/norm
      
    m=v.dot(newS).dot(u_t)
    #pongo randomicamente dei valori negativi
    for i in range(np.random.randint(0,m.shape[0]*m.shape[1]-1)):
      #activation_output scelgo  una valore casuale e cambio il segno
      randRow=np.random.randint(0,m.shape[0])
      randCol=np.random.randint(0,m.shape[1])
      m[randRow,randCol]=-m[randRow,randCol]
    return m
     
  def __theano_build__(self):
    W,W_in,W_out,W_fb=self.W,self.W_in,self.W_out,self.W_fb
    u=T.dmatrix('u')
    
  
    def forward_prop_step(u_t,x_t_prev, W_in, W, W_out):
	x_t = self.activation_function(T.dot(u_t,W_in)+T.dot(x_t_prev,W))
	o_t = self.activation_output(T.dot(x_t,W_out))
	return [o_t, x_t]
      
    [o,x], updates = theano.scan(
            forward_prop_step,
            sequences=u,
            outputs_info=[None, dict(initial=T.zeros(self.resevoir_dim))],
            non_sequences=[W_in, W, W_out])
    
    output=o
    resevoir=x
    self.computeOutput = theano.function([u], output,allow_input_downcast=True)
    self.computeResevoir = theano.function([u], resevoir,allow_input_downcast=True)
    y=T.dmatrix(name='y')
    def mse(t, y):
      # error between output and target
      return T.mean((t - y) ** 2)
    
    o_error = T.sum(mse(o, y))
    self.dWout = T.grad(o_error, W_out)
    
    self.bptt = theano.function([u, y], [self.dWout],allow_input_downcast=True)
    learning_rate = T.scalar('learning_rate')
    
    self.sgd_step = theano.function([u,y,learning_rate], [], 
                      updates=[(self.W_out, self.W_out - learning_rate * self.dWout)],allow_input_downcast=True)
  
  def TrainESN(self,inputSet,targetSet):

    #self.__theano_build__()
    proj=self.computeResevoir(inputSet[0])
    targetMatrix=targetSet[0]
    for seq,tar in zip(inputSet[1::],targetSet[1::]):
      proj=np.concatenate((proj,self.computeResevoir(seq)))
      targetMatrix=np.concatenate((targetMatrix,tar))
    self.W_out.set_value(np.linalg.pinv(proj).dot(targetMatrix))
  
  def OnlineTrain(self,inputSet,targetSet,learning_rate):
    #print "trainingFun"
    #print targetSet
    #raw_input("--")
    for inSeq,tarSeq in zip(inputSet,targetSet):
      self.sgd_step(inSeq,tarSeq,learning_rate)
      
      
  def computeOut(self,inputSet):
    output=[]
    for seq in inputSet:

      output.append(self.computeOutput(seq)[-1][0]) 
    #print "outFun"
    #print output
    #raw_input("----")
    #print self.W_out.get_value()
    #raw_input("---------------------")
    return output
      
    

def main():
  S1=np.array([[0,0,1],[0,1,0],[1,0,0],[1,1,1],[1,0,1],[0,1,0],[1,0,0],[1,1,1]])
  S2=np.array([[0,1,0],[1,0,0],[1,1,1]])
  S3=np.array([[1,1,1],[1,0,1],[0,1,0],[1,0,1]])
  #T1=np.array([[0,0,1,1],[0,1,1,0],[1,1,0,0],[1,1,1,1],[1,1,0,1],[0,1,1,0],[1,1,0,0],[1,1,1,1]])
  #T2=np.array([[0,1,0,1],[1,0,0,1],[1,1,1,1]])
  #T3=np.array([[1,1,1,1],[1,1,0,1],[0,1,1,0],[1,0,1,1]])
  
  T1=np.array([[1],[1],[1],[1],[1],[1],[1],[1]])
  T2=np.array([[0],[0],[0]])
  T3=np.array([[1],[1],[0],[1]])
  ##------------test DS------------#
  dataset=[S1,S2,S3]
  targetSet=[T1,T2,T3]
  maxLen=5
  nSample=3
  sampleLen=[5,3,4]
  singleSampleDim=3
  nOutUnits=1
  nComponet=3
  dimGroup=2
  nEpoch=10
  learningRate=1
  
  
  net= EchoStateNetwork(singleSampleDim,nComponet,nOutUnits,T.nnet.sigmoid,lambda x:x)
  


  #net.TrainESN(dataset,targetSet)
  #print net.computeOut(dataset)
  #print net.W_out.get_value()

  print "!---!"
  net= EchoStateNetwork(singleSampleDim,nComponet,nOutUnits,T.nnet.sigmoid,lambda x:x)
  #print net.W_out.get_value()
  net.OnlineTrain(dataset[0:2],targetSet[0:2],learningRate)
  #raw_input()
  #print net.W_out.get_value()
  net.OnlineTrain(dataset[2::],targetSet[2::],learningRate)
  #print net.computeOut(dataset)
  #print resevoir
  print "Test"
  
  
if __name__ == '__main__':
  main()
