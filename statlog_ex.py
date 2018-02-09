import numpy as np
import random, math

from sklearn.linear_model.stochastic_gradient import SGDClassifier
from frameworks.SelfLearning import SelfLearningModel
import sklearn.svm
from plot_utils import evaluate

import datasets
from methods import scikitTSVM


# load data
dataset_path = "./p_datasets"
dataset =dataset_path + "/heart.dat"
hearts = datasets.standardize_hearts(dataset)


X = hearts['featureset']
ytrue = hearts['target']

# print(len(ytrue))
# label a few points
perc = .20
labeled_N = math.floor(len(ytrue) * perc)

nsamples = math.floor(labeled_N/2)
ys = np.array([-1]*len(ytrue)) # -1 denotes unlabeled point
random_labeled_points = list(np.random.choice(np.where(ytrue == 0)[0], int(nsamples)))+ \
                        list(np.random.choice(np.where(ytrue == 1)[0], int(nsamples)))

ys[random_labeled_points] = ytrue[random_labeled_points]

# supervised score
# basemodel = SGDClassifier(loss='log', penalty='l1', tol=1e-3, max_iter=1000) # scikit logistic regression
# basemodel.fit(X[random_labeled_points, :], ys[random_labeled_points])
# print ("supervised log.reg. score", basemodel.score(X, ytrue))
kernel = "rbf"
Xsupervised = X[ys!=-1, :]
ysupervised = ys[ys!=-1]

lbl = "Purely supervised SVM:"
print (lbl)
basemodel = sklearn.svm.SVC(kernel=kernel, probability=True)
basemodel.fit(Xsupervised, ysupervised)
evaluate(basemodel, X, ys, ytrue, lbl)

#
# fast (but naive, unsafe) self learning framework
ssmodel = SelfLearningModel(basemodel)
ssmodel.fit(X, ys)
print ("self-learning log.reg. score", ssmodel.score(X, ytrue))

Xsupervised = X[ys!=-1, :]
ysupervised = ys[ys!=-1]

lbl = "Purely supervised SVM:"
print (lbl)
model = sklearn.svm.SVC(kernel=kernel, probability=True)
model.fit(Xsupervised, ysupervised)
evaluate(model, X, ys, ytrue, lbl)

lbl =  "S3VM (Gieseke et al. 2012):"
print (lbl)
model = scikitTSVM.SKTSVM(kernel=kernel)
model.fit(X, ys)
evaluate(model, X, ys, ytrue, lbl)





