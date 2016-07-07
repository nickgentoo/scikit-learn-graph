import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '',''))
import numpy as np


#"sys.path.append('..\\..\\Multiple Kernel Learning\\Framework')"
if len(sys.argv)<3:
    sys.exit("python normalize_stability.py eigenvalues_file Stability_file outfile")


eigen=open(sys.argv[1],'r')
stab=open(sys.argv[2],'r')
e=float(eigen.readlines()[0])

temp=stab.readlines()[-1].strip().split(" ")
#print temp
s=float(temp[5])
w=float(temp[7])
print e 
print s, w
w_norm=w/e
output=open(sys.argv[3],'w')
output.write("Stability "+str(s)+" W_max "+str(w)+ " W_max_norm "+str(w_norm))
output.close()
eigen.close()
stab.close()
