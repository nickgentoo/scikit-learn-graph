AUTHOR: Michele Donini
EMAIL: mdonini@math.unipd.it
SITE: http://www.math.unipd.it/~mdonini

This is the code for EasyMKL, a multiple learning algorithm from the papers:
- EasyMKL: a scalable multiple kernel learning algorithm by Fabio Aiolli and Michele Donini
- Easy multiple kernel learning by Fabio Aiolli and Michele Donini

Link: http://www.math.unipd.it/~mdonini/publications.html


Files in the archive:
- EasyMKL.py contains the EasyMKL implementation
- komd.py contains the KOMD kernel machine implementation 
	(for more information see: A Kernel Method for the Optimization of the Margin Distribution 
	by F. Aiolli, G. Da San Martino, and A. Sperduti)
- toytest_EasyMKL.py is the toytest in order to learn how EasyMKL works


Very short example:
# Given the kernels list evaluated among the training examples:
# 	kernel_list_trainxtrain (a list of kernels of dimension: train_set * train_set)
# TRAIN:
from EasyMKL import EasyMKL
l = 0.5 # lambda
easy = EasyMKL(lam=l, tracenorm = True)
easy.train(kernel_list_trainxtrain,Ytr)
# Given the kernels list evaluated among the training and test:
# 	kernel_list_testxtrain (a list of kernels of dimension: test_set * train_set)
# TEST with "y_real_test" as correct labels:
from sklearn.metrics import roc_auc_score
ranktest = np.array(easy.rank(kernel_list_testxtrain))
print roc_auc_score(y_real_test,ranktest)



If you use this code, please cite:

@article{aiolli2015easymkl,
  title={EasyMKL: a scalable multiple kernel learning algorithm},
  author={Aiolli, Fabio and Donini, Michele},
  journal={Neurocomputing},
  year={2015},
  publisher={Elsevier}
}


Thank you,
Michele Donini.