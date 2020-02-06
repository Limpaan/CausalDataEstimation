import numpy as np
import pathlib


def import_data():
    path = pathlib.Path(__file__).parent.absolute()
    path = str(path) + "\\" + "Datasets" + "\\"
    train = np.load(path + "\\" + "ihdp_npci_1-100.train.npz")
    test = np.load(path + "\\" + "ihdp_npci_1-100.test.npz")
    for key in train.keys():
        print(key)
        print(train[key].shape)
    return {'train': train, 'test': test}

import_data()