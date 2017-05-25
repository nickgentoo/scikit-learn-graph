import sys
sys.path.insert(0, '../')
import numpy as np

def save(dataset,fileName):
  #lo rendo un numpy array per facilitare il salvataggio
  saveData=np.array(dataset)
  np.save("../../../../../cluster3/lpasa/linSysNet_ProjectedData/Group/"+fileName , saveData)
  
  
def load(fileName):
  data=np.load("../../../../../cluster3/lpasa/linSysNet_ProjectedData/Group/"+fileName+".npy")
  res=[]
  for seq in data:
     res.append(seq[0])
  return res 

if __name__=="__main__":
  
  tar=np.array([[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]])
  gen=np.array([[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]])
  dataset=[tar,gen]
  #save(dataset,"prova")
  a=load("BaseTest_Train")
  
  print type(a[0])