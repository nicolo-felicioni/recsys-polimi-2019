import scipy.sparse as sps
import numpy as np
import time
import random
import os


class Splitter(object):

    def __init__(self, urm):
        self.urm = urm


    # Split a scipy csr matrix in two matrixes:
    #   - train set csr matrix
    #   - test set csr matrix
    # In particular:
    #   - for each row, at least one element of the row will be in the test set
    #   - the number of elements in the test will converge to (n_elements * probability + n_rows)

    # Parameter:
    #   - csr_matrix: the input csr matrix
    #   - random_seed: the seed for generating random numbers
    #   - probability: the probability that an element is copied to the test set
    #   - threshold: rows with less than this amount, will have only one element in the test set

    def _split_train_test(self, random_seed=16, threshold=5,
                                         probability=0, debug_print=True, exclude_target_users_from_test=True,
                                         exclude_cold_users_from_test=True, k=1):
        # Data structures
        urm = self.urm
        #
        #
        # Number of rows in the input matrix
        # in URM, it is the equivalent to users
        n_rows = urm.shape[0]
        # The data contained in the input matrix
        # it is an array '1', each one representing an interaction
        data = urm.data
        # The indices in the input matrix
        # it represents the columns containing interactions
        indices = urm.indices
        # The indice pointers in the input matrix
        # it represents where a new row starts
        indptr = urm.indptr
        #
        #
        cold_users = []

        # Auxiliary data structures needed in order to build the csr train matrix
        train_data = []
        train_user = []
        train_item = []

        # Auxiliary data structures needed in order to build the csr train matrix
        test_data = []
        test_user = []
        test_item = []

        # Initializing the random seed
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Monitoring time needed for executing the split
        starting_time = time.time()
        if debug_print:
            print("Starting to split the CSR matrix in train and test csr matrixes...")

        # For each row
        # equivalent to: for each user
        for user_id in range(0, n_rows):
            if user_id not in cold_users:
                max_elements = min(k, len(urm[user_id].indices))
                k_elements = random.sample(list(urm[user_id].indices), max_elements)
                for item_id in k_elements:
                    test_data.append(1)
                    test_user.append(user_id)
                    test_item.append(item_id)
                if len(urm[user_id].indices) > threshold:
                    for item_id in urm[user_id].indices:
                        if item_id not in k_elements:
                            if np.random.random(1) < probability:
                                test_data.append(1)
                                test_user.append(user_id)
                                test_item.append(item_id)
                            else:
                                train_data.append(1)
                                train_user.append(user_id)
                                train_item.append(item_id)
                else:
                    for item_id in urm[user_id].indices:
                        if item_id not in k_elements:
                            train_data.append(1)
                            train_user.append(user_id)
                            train_item.append(item_id)

        # Compute the CSR matrix for the train set
        train_csr = sps.csr_matrix((train_data, (train_user, train_item)), shape=urm.shape)
        # Compute the CSR matrix for the test set
        test_csr = sps.csr_matrix((test_data, (test_user, test_item)), shape=urm.shape)

        return train_csr, train_user, train_item, test_csr, test_user, test_item


    def split_train_test_check_if_stored(self, random_seed=16, threshold=5,
                         probability=0, debug_print=True, exclude_target_users_from_test=True,
                         exclude_cold_users_from_test=True, k=1):
        split_path = "stored_matrices/split"
        path = split_path + "/" + str(random_seed)

        if(os.path.exists(path)):
            print("Loading data from files stored...")
            # Load the CSR matrix for the train set
            train_csr = sps.load_npz(path + "/" + "train_csr.npz")
            train_user = self._loadList(path + "/" + "train_user.npy")
            train_item = self._loadList(path + "/" + "train_item.npy")

            # Load the CSR matrix for the test set
            test_csr = sps.load_npz(path + "/" + "test_csr.npz")
            test_user = self._loadList(path + "/" + "test_user.npy")
            test_item = self._loadList(path + "/" + "test_item.npy")

        else:
            print("Can't load data. Reading data...")
            train_csr, train_user, train_item, test_csr, test_user, test_item = self._split_train_test(random_seed=random_seed, threshold=threshold,
                         probability=probability, debug_print=debug_print, exclude_target_users_from_test=exclude_target_users_from_test,
                         exclude_cold_users_from_test=exclude_cold_users_from_test, k=k)

            # create the directory
            os.mkdir(path)

            # Save the CSR matrix for the train set
            sps.save_npz(path + "/" + "train_csr", train_csr)
            self._saveList(path + "/" + "train_user", train_user)
            self._saveList(path + "/" + "train_item", train_item)

            # Save the CSR matrix for the test set
            sps.save_npz(path + "/" + "test_csr", test_csr)
            self._saveList(path + "/" + "test_user", test_user)
            self._saveList(path + "/" + "test_item", test_item)

        self.train_csr = train_csr
        self.test_csr = test_csr
        self.ids_warm_train_users = np.unique(train_user)
        self.ids_warm_train_items = np.unique(train_item)
        self.ids_cold_train_users = np.array([user for user in test_user if user not in self.ids_warm_train_users])
        self.ids_cold_train_items = np.array([item for item in test_item if item not in self.ids_warm_train_items])
        self.number_of_warm_train_users = self.ids_warm_train_users.shape[0]
        self.number_of_warm_train_items = self.ids_warm_train_items.shape[0]
        self.number_of_cold_train_users = self.ids_cold_train_users.shape[0]
        self.number_of_cold_train_items = self.ids_cold_train_items.shape[0]

    def _saveList(self, filename, myList):
        # the filename should mention the extension 'npy'
        np.save(filename, myList)
        print("Saved successfully!")

    def _loadList(self, filename):
        # the filename should mention the extension 'npy'
        tempNumpyArray = np.load(filename)
        return tempNumpyArray.tolist()
