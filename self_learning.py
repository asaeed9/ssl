import matplotlib
import sklearn
import numpy as np
from random import random

from frameworks.SelfLearning import *

# load `Lung cancer' dataset from mldata.org
cancer = fetch_mldata("Lung cancer (Ontario)")
X = cancer.target.T
ytrue = np.copy(cancer.data).flatten()
ytrue[ytrue>0]=1

# label a few points
labeled_N = 4
ys = np.array([-1]*len(ytrue)) # -1 denotes unlabeled point
random_labeled_points = random.sample(np.where(ytrue == 0)[0], labeled_N/2)+\
                        random.sample(np.where(ytrue == 1)[0], labeled_N/2)
ys[random_labeled_points] = ytrue[random_labeled_points]

# supervised score
basemodel = SGDClassifier(loss='log', penalty='l1') # scikit logistic regression
basemodel.fit(X[random_labeled_points, :], ys[random_labeled_points])
print ("supervised log.reg. score", basemodel.score(X, ytrue))

# fast (but naive, unsafe) self learning framework
ssmodel = SelfLearningModel(basemodel)
ssmodel.fit(X, ys)
print ("self-learning log.reg. score", ssmodel.score(X, ytrue))

# semi-supervised score (base model has to be able to take weighted samples)
# ssmodel = CPLELearningModel(basemodel)
# ssmodel.fit(X, ys)
# print "CPLE semi-supervised log.reg. score", ssmodel.score(X, ytrue)
#
# # semi-supervised score, RBF SVM model
# ssmodel = CPLELearningModel(sklearn.svm.SVC(kernel="rbf", probability=True), predict_from_probabilities=True) # RBF SVM
# ssmodel.fit(X, ys)
# print "CPLE semi-supervised RBF SVM score", ssmodel.score(X, ytrue)

# supervised log.reg. score 0.410256410256
# self-learning log.reg. score 0.461538461538
# semi-supervised log.reg. score 0.615384615385
# semi-supervised RBF SVM score 0.769230769231