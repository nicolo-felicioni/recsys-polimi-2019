import random as rnd
import time

import scipy.sparse as sps
import numpy as np

from DataObject import DataObject
from DataReader import DataReader


# Example of usage
# split(urm,
#       [(0, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 160), (161, 320), (321, 640)],
#       [0, 1, 2, 3, 4, 5, 6, 7],
#       [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
def split(urm: sps.csr_matrix,
          range_intervals_list: list,
          k_out_list: list,
          probability_list: list) -> (sps.csr_matrix, sps.csr_matrix):
    # Initialize timer
    start_time = time.time()

    # Checking the size of the list
    assert len(range_intervals_list) == len(k_out_list) == len(probability_list), \
        "The input lists must have the same length"

    # Generating a copy of the matrix
    matrix = urm.tocsr(copy=True)

    # Initializing the matrices
    train_matrix = sps.lil_matrix(matrix.shape, dtype=np.float)
    test_matrix = sps.lil_matrix(matrix.shape, dtype=np.float)

    # Users
    users = range(matrix.shape[0])

    # Creating a list of triple (range_interval, k_out, probability)
    triple_list = [(r, k, p) for r, k, p in zip(range_intervals_list, k_out_list, probability_list)]

    # Start splitting the input matrix
    for user in users:

        # Finding the items with which the user has interacted
        items = matrix[user].indices
        n_items = len(items)

        # Finding the splitting category for the user
        k, p = -1, -1
        for range_interval, k_out, probability in triple_list:
            if range_interval[0] <= n_items <= range_interval[1]:
                k = k_out
                p = probability
                break

        # Extracting k unique items (if possible)
        n_items_to_be_extracted = min(k, n_items)
        rnd.shuffle(items)
        k_extracted_items = items[:n_items_to_be_extracted]
        remaining_items = items[n_items_to_be_extracted:]

        # Extracting elements with probability p
        p_extracted_items = [item for item in remaining_items if rnd.random() <= p]
        remaining_items = [item for item in remaining_items if item not in p_extracted_items]

        # Incrementally generate fill the train and test matrix
        for item in remaining_items:
            train_matrix[user, item] = matrix[user, item]
        for item in k_extracted_items:
            test_matrix[user, item] = matrix[user, item]
        for item in p_extracted_items:
            test_matrix[user, item] = matrix[user, item]

    # Converting train matrix and test matrix to csr format
    train_matrix = train_matrix.tocsr()
    test_matrix = test_matrix.tocsr()

    # Checking output matrices
    assert train_matrix.shape == test_matrix.shape == matrix.shape, \
        "Train and test matrices should have the same shape as the input one"
    assert (train_matrix.nnz + test_matrix.nnz) == matrix.nnz, \
        f"The sum of train and test matrices nnz elements should be exactly the nnz elements in the input matrix:\n" \
        f"train matrix nnz elements : {train_matrix.nnz}\n" \
        f"test matrix nnz elements : {test_matrix.nnz}\n" \
        f"matrix nnz elements : {matrix.nnz}\n"

    # Ending time
    end_time = time.time()
    print(f"Time taken to split the input matrix : {end_time - start_time} seconds")

    return train_matrix, test_matrix


def split_with_triple(urm: sps.csr_matrix,
                      triple_list: list):
    triple_list = np.array(triple_list)
    range_intervals = triple_list[:, 0]
    leave_k_out = triple_list[:, 1]
    probability = triple_list[:, 2]
    return split(urm,
                 range_intervals_list=range_intervals,
                 k_out_list=leave_k_out,
                 probability_list=probability)
