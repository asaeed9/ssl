import numpy as np
import random, math

from sklearn.linear_model.stochastic_gradient import SGDClassifier
from frameworks.SelfLearning import SelfLearningModel
import sklearn.svm
from plot_utils import evaluate

import matplotlib.pyplot as plt

import datasets
from methods import scikitTSVM

# Series Statistics: Utilities
def mean_sd_se(list_of_values):
    print(list_of_values)
    mean = sum(list_of_values) / max(1, len(list_of_values))
    var = sum([(value - mean) * (value - mean) for value in list_of_values]) / max(1, len(list_of_values))
    std_dev = math.sqrt(var)
    std_err = std_dev / math.sqrt(max(1, len(list_of_values)))
    return (mean, std_dev, std_err)


def series_statistics(list_of_lists):
    print(list_of_lists)
    means = []
    std_devs = []
    std_errs = []
    for list_of_values in list_of_lists:
        (mean, sd, se) = mean_sd_se(list_of_values)
        means.append(mean)
        std_devs.append(sd)
        std_errs.append(se)
    return (means, std_devs, std_errs)


# load data
dataset_path = "./p_datasets"
dataset = dataset_path + "/pendigits.tra"
pendigits = datasets.standardize_pendigits(dataset)

X = pendigits['featureset']
ytrue = pendigits['target']

print(len(ytrue))

def average(arr, iters, trials):
    avg = []
    for i in range(trials):
        avg.append(np.mean(arr[i * iters:i * iters + iters]))
    return avg


percents = [0.01, 0.05, 0.1, .3, .5, 1]
actual_amount = []
sgd_perf = []
svm_perf = []
self_learning_perf = []
s3vm_perf = []

iters = 5

for per in percents:

    # label a few points
    perc = per
    labeled_N = math.floor(len(ytrue) * perc)
    print(labeled_N)
    actual_amount.append(labeled_N)

    sgd_active = []
    svm_active = []
    self_learning_active = []
    s3vm_active = []

    for it in range(iters):
        nsamples = math.floor(labeled_N / 2)
        ys = np.array([-1] * len(ytrue))  # -1 denotes unlabeled point
        random_labeled_points = list(np.random.choice(np.where(ytrue == 0)[0], int(nsamples))) + \
                                list(np.random.choice(np.where(ytrue == 1)[0], int(nsamples)))

        ys[random_labeled_points] = ytrue[random_labeled_points]

        # supervised score
        basemodel = SGDClassifier(loss='hinge', penalty='l1', tol=1e-3, max_iter=1000)  # scikit logistic regression
        basemodel.fit(X[random_labeled_points, :], ys[random_labeled_points])

        sgd_active.append(basemodel.score(X, ytrue))

        kernel = "rbf"

        svm_model = sklearn.svm.SVC(kernel=kernel, probability=True)
        ssmodel = SelfLearningModel(svm_model)
        ssmodel.fit(X, ys)
        self_learning_active.append(ssmodel.score(X, ytrue))

        Xsupervised = X[ys != -1, :]
        ysupervised = ys[ys != -1]

        lbl = "Purely supervised SVM:"
        model = sklearn.svm.SVC(kernel=kernel, probability=True)
        model.fit(Xsupervised, ysupervised)
        acc = evaluate(model, X, ys, ytrue, lbl)
        svm_active.append(acc)

        lbl = "S3VM (Gieseke et al. 2012):"
        model = scikitTSVM.SKTSVM(kernel=kernel)
        model.fit(X, ys)
        acc = evaluate(model, X, ys, ytrue, lbl)
        s3vm_active.append(acc)

    sgd_perf.append(sgd_active)
    self_learning_perf.append(self_learning_active)
    svm_perf.append(svm_active)
    s3vm_perf.append(s3vm_active)

(means, std_devs, std_errs) = series_statistics(sgd_perf)
(sl_means, sl_std_devs, sl_std_errs) = series_statistics(self_learning_perf)
(svm_means, svm_std_devs, svm_std_errs) = series_statistics(svm_perf)
(s3vm_means, s3vm_std_devs, s3vm_std_errs) = series_statistics(s3vm_perf)

plt.errorbar(percents, means, yerr=std_devs, fmt="c", label="SGD")

plt.errorbar(percents, sl_means, yerr=sl_std_devs, fmt="b", label="SVM RBF Self Training")
plt.errorbar(percents, svm_means, yerr=svm_std_devs, fmt="r", label="SVM RBF")
plt.errorbar(percents, s3vm_means, yerr=s3vm_std_devs, fmt="m", label="S3VM")
plt.xlabel("Percent of dataset")
plt.ylabel("Transduction Accuracy (%)")
plt.title("Baselines on mldata Heart")
plt.legend(loc="lower right")
plt.show()