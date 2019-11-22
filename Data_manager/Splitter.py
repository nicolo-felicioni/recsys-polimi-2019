import scipy.sparse as sps
import numpy as np
import time
import random


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

    def split_train_test(self, random_seed=16, threshold=5,
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
        self.train_csr = train_csr
        self.test_csr = test_csr
        self.ids_warm_train_users = np.unique(train_user)
        self.ids_warm_train_items = np.unique(train_item)
        self.ids_cold_train_users = np.array([user for user in test_user if user not in self.ids_warm_train_users])
        self.ids_cold_train_items = np.array([item for item in test_item if item not in self.ids_warm_train_items])
        self.number_of_warm_train_users = self.ids_warm_train_users.shape[0]
        self.number_of_warm_train_items = self.ids_warm_train_items.shape[0]
        self.number_of_cold_train_users = self.ids_cold_train_users.shape[0]
        self.number_of_cold_train_items = self.ids_cold_train_users.shape[0]
