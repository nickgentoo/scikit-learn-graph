import numpy as np
import numpy.linalg as linalg

def ECPMatrix(m,epsilon=0.03):
    #si calcola svd, si fa si che tutti gli autovalori siano inferriori a 1 e si richrea la matrice
    v,s,u_t=linalg.svd(m)
    norm=s[0]+epsilon
    newS=np.zeros((v.shape[1],u_t.shape[0]))
    for i,element in enumerate(s):
      newS[i,i]=element/norm
      
    m=v.dot(newS).dot(u_t)
    #pongo randomicamente dei valori negativi
    for i in range(np.random.randint(0,m.shape[0]*m.shape[1]-1)):
      #sactivation_outputcelgo  una valore casuale e cambio il segno
      randRow=np.random.randint(0,m.shape[0])
      randCol=np.random.randint(0,m.shape[1])
      m[randRow,randCol]=-m[randRow,randCol]
    return m


def createEPCMatrix(nrows,ncols,scale=1):
  return ECPMatrix(np.random.uniform(low=-scale, high=scale,size=(nrows,ncols))) 


def main():
  m=createEPCMatrix(5,3)
  print m.shape

if __name__ == '__main__':
  main()