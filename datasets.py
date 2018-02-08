from sklearn.datasets import fetch_mldata
import numpy as np
import math

global inmemory_dataset
global labelled_n

def standardize_hearts(ds_path_to_file):
    global inmemory_dataset

    data = np.genfromtxt(ds_path_to_file,
                         skip_header=1,
                         skip_footer=1,
                         names=True,
                         dtype=None,
                         delimiter=' '
                         )
    inmemory_dataset = data
    inmemory_dataset = np.asarray([list(inmemory_dataset[vec, ]) for vec in range(len(inmemory_dataset))])

    target = inmemory_dataset[:, -1]
    featureset = inmemory_dataset[:, 0:-1]

    target[target == 1] = 0
    target[target == 2] = 1

    dataset = {"featureset": featureset, "target":target}

    return dataset

def standardize_pendigits(ds_path_to_file):

    global inmemory_dataset

    data = np.genfromtxt(ds_path_to_file,
                         skip_header=1,
                         skip_footer=1,
                         names=True,
                         dtype=None,
                         delimiter=','
                         )
    inmemory_dataset = data
    inmemory_dataset = np.asarray([list(inmemory_dataset[vec, ]) for vec in range(len(inmemory_dataset))])

    target = inmemory_dataset[:, -1]
    featureset = inmemory_dataset[:, 0:-1]

    target[target == 1] = 0
    target[target == 2] = 1

    dataset = {"featureset": featureset, "target":target}

    return dataset

def standardize_usps(ds_path_to_file):
    inmem_dataset = fetch_mldata(ds_path_to_file)
    dataset = {"featureset": inmem_dataset['data'], "target": inmem_dataset['target']}

    return dataset



def divide_response_predictor(isdat = False, response = -1):
    global inmemory_dataset

    if isdat:
        inmemory_dataset = np.asarray([list(inmemory_dataset[vec, ]) for vec in range(len(inmemory_dataset))])

    target = inmemory_dataset[:, -1]
    featureset = inmemory_dataset[:, 0:-1]

    target[target == 1] = 0
    target[target == 2] = 1

    return featureset, target

def sample_labelled_data(featureset, target, labelled_n = 20):
    unlabelled_ys = np.array([-1] * len(target))  # -1 denotes unlabeled point

    nsamples = math.floor(labelled_n/2.)

    random_labeled_points = list(np.random.choice(np.where(target == 0)[0], int(nsamples))) + \
                            list(np.random.choice(np.where(target == 1)[0], int(nsamples)))

    unlabelled_ys[random_labeled_points] = target[random_labeled_points]

    print(unlabelled_ys[unlabelled_ys != -1])

    return featureset[random_labeled_points, :], unlabelled_ys









