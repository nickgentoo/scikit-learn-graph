__author__ = "Carlo Maria Massimo"

import sys
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from skgraph.datasets import load_graph_datasets

if __name__=='__main__':
    if len(sys.argv)<2:
        sys.exit("python SVM_precomputed.py precomputed_gram_file dataset impl")
    
    gram_file = str(sys.argv[1])
    dataset = str(sys.argv[2])

    if len(sys.argv) > 3:
        impl = str(sys.argv[3])
    else:
        impl = "skgraph"

    # skgraph
    if impl == "eden":
        gram = np.loadtxt(gram_file)
        new_gram = gram
    else:
        gram, tr_y = load_svmlight_file(gram_file)
        new_gram = gram[:, 1:gram.shape[1]].todense()

    # EDeN

    if dataset == 'CAS':

        ds = load_graph_datasets.load_graphs_bursi()

    elif dataset == 'CPDB':
        ds = load_graph_datasets.load_graphs_CPDB()
    elif dataset == 'AIDS':
        ds = load_graph_datasets.load_graphs_AIDS()
    elif dataset == 'NCI1':
        ds = load_graph_datasets.load_graphs_NCI1()

    y = ds.target

#    print gram_file

    clf = svm.SVC(C=10, kernel='precomputed')
    #clf.fit(new_gram, y)

    scores = cross_validation.cross_val_score(clf, new_gram, y, cv=10)

    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    """
    #induce a predictive model
    from sklearn.linear_model import SGDClassifier
    predictor = SGDClassifier()
    
    from sklearn import cross_validation
    scores = cross_validation.cross_val_score(predictor, X, y,cv=10, scoring='accuracy')
    
    import numpy as np
    print('Accuracy: %.4f +- %.4f' % (np.mean(scores),np.std(scores)))
    
    scores = cross_validation.cross_val_score(predictor, X, y,cv=10, scoring='roc_auc')
    
    print('AUC ROC: %.4f +- %.4f' % (np.mean(scores),np.std(scores)))
    """


#res = {}
#
#for g in g_it.graphs:
#    tmp = ODDkernel.getFeaturesNoCollisionsExplicit(g).items()
#
#    for (k,v) in tmp:
#        if res.get(k) == None:
#            res[k] = v
#        else:
#            res[k] += v
#
#
#plt.plot(res.values()[:100])
