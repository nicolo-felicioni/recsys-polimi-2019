import threading
from threading import Thread

import numpy as np
from multiprocessing import Pool
import ml_metrics
import sklearn
from joblib import Parallel, delayed
import multiprocessing
import ml_metrics


def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score

def calc(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    return precision_score, recall_score, map_score

# We pass as paramether the recommender class

def evaluate_algorithm(test, users, recommender_object, at=10, remove_top=0):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0
    userList_unique = users
    URM_test = test
    num_eval = len(userList_unique)

    for user_id in userList_unique:

        relevant_items = URM_test[user_id].indices
        if len(relevant_items):
            recommended_items = recommender_object.recommend(user_id, cutoff=(at + remove_top))[remove_top:at+remove_top]
            if len(recommended_items):
                _m = MAP(recommended_items, relevant_items)
                # cumulative_precision += _prec
                # cumulative_recall += _rec
                cumulative_MAP += _m

    # cumulative_precision /= num_eval
    # cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    return("Recommender performance is: MAP = {:.8f}".format(cumulative_MAP)), cumulative_MAP

def evaluate_algorithm_parallel(test, users, recommender_object, at=10, remove_top=0, parallelism=4):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0
    userList_unique = users
    URM_test = test
    n = users.shape[0]
    #cpu_count = multiprocessing.cpu_count()
    cpu_count=parallelism

    # mutithreading version
    def eval_single_user(_user_id, _test=test, _at=at, _remove_top=remove_top):
        cumulative_precision = 0.0
        cumulative_recall = 0.0
        cumulative_MAP = 0.0
        relevant_items = URM_test[_user_id].indices
        if len(relevant_items):
            recommended_items = recommender_object.recommend(_user_id, cutoff=(_at + _remove_top))[_remove_top:_at + _remove_top]
            if len(recommended_items):
                cumulative_precision += precision(recommended_items, relevant_items)
                cumulative_recall += recall(recommended_items, relevant_items)
                cumulative_MAP += MAP(recommended_items, relevant_items)
        return cumulative_precision, cumulative_recall, cumulative_MAP

    results = Parallel(n_jobs=cpu_count)(delayed(eval_single_user)(user_id, _test=test, _at=at, _remove_top=remove_top) for user_id in userList_unique)
    for result in results:
        cumulative_precision += result[0]
        cumulative_recall += result[1]
        cumulative_MAP += result[2]

    cumulative_precision = cumulative_precision/n
    cumulative_recall = cumulative_recall/n
    cumulative_MAP = cumulative_MAP/n

    return("Recommender performance is: Precision = {:.8f}, Recall = {:.8f}, MAP = {:.8f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP)), cumulative_MAP