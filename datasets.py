from sklearn.datasets import fetch_mldata
import numpy as np
import math

global inmemory_dataset
global labelled_n

def pick_toptwo_labels(inmem_dataset):
    ytrue = inmem_dataset['target']
    featureset = inmem_dataset['data']

    uniq_target = set(ytrue)
    max_label_dic = {}
    if (len(uniq_target) > 2):
        for val in uniq_target:
            max_label_dic[val] = len(ytrue[ytrue == val])

    top_2 = sorted(max_label_dic, key=max_label_dic.get, reverse=True)[:2]
    print(max_label_dic)

    featureset = featureset[np.where((ytrue == top_2[0]) | (ytrue == top_2[1]))]
    target = ytrue[np.where((ytrue == top_2[0]) | (ytrue == top_2[1]))]

    # target[target == top_2[0]] = 0
    # target[target == top_2[1]] = 1

    return featureset, target


def pick_any2_labels(inmem_dataset, labels_to_keep=[1,2]):
    ytrue = inmem_dataset['target']
    featureset = inmem_dataset['data']

    uniq_target = set(ytrue)
    # print(uniq_target)

    if len(labels_to_keep) != 2:
        print("The function only supports two variables...")
        return

    featureset = featureset[np.where((ytrue == labels_to_keep[0]) | (ytrue == labels_to_keep[1]))]

    target = ytrue[np.where((ytrue == labels_to_keep[0]) | (ytrue == labels_to_keep[1]))]

    print(set(target))

    target[target == labels_to_keep[0]] = 0
    target[target == labels_to_keep[1]] = 1

    return featureset, target


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

def standardize_pendigits(ds_path_to_file, labels_to_keep=[1,2]):

    # global inmemory_dataset
    data = np.genfromtxt(ds_path_to_file,
                         skip_header=1,
                         skip_footer=1,
                         names=True,
                         dtype=None,
                         delimiter=','
                         )
    inmem_dataset = data
    inmem_dataset = np.asarray([list(inmem_dataset[vec, ]) for vec in range(len(inmem_dataset))])

    dataset = {"data":inmem_dataset[:, 0:-1], "target":inmem_dataset[:, -1]}

    featureset, target = pick_any2_labels(dataset, labels_to_keep)

    dataset = {"featureset": featureset, "target":target}

    return dataset

def standardize_usps(ds_path_to_file, labels_to_keep=[1,2]):
    inmem_dataset = fetch_mldata(ds_path_to_file)


    # featureset, target = pick_toptwo_labels(inmem_dataset)

    print(set(inmem_dataset['target']))

    featureset, target = pick_any2_labels(inmem_dataset, labels_to_keep)

    dataset = {"featureset": featureset, "target": target}

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









