import numpy as np
import pathlib


class Importer:

    def __init__(self, folder="Datasets", train_name="ihdp_npci_1-100.train.npz", test_name="ihdp_npci_1-100.test.npz"):
        path = pathlib.Path(__file__).parent.absolute()
        path = str(path) + "\\" + folder + "\\"
        self.train = np.load(path + "\\" + train_name)
        self.test = np.load(path + "\\" + test_name)
        for key in self.train.keys():
            assert key in self.test.keys()

    def get_training_set(self):
        return self.train

    def get_test_set(self):
        return self.test

    def print_keys(self, shape=True):
        for key in self.train.keys():
            print(key)
            if shape:
                print(self.train[key].shape)