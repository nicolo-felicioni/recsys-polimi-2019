#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender_utils import check_matrix
from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
from heapq import *

from Base.IR_feature_weighting import okapi_BM_25, TF_IDF

from Base.Similarity.Compute_Similarity import Compute_Similarity
import pandas as pd
from random import seed
from random import random
import numpy as np
import scipy.sparse as sps

from DataObject import DataObject


class UserKNNAgeRecommender(BaseSimilarityMatrixRecommender):
    """ UserKNN Content Based recommender
        It uses the age information given by the dataset.
    """


    RECOMMENDER_NAME = "UserKNNAgeRecommender"

    def __init__(self, data: DataObject, df_age: pd.DataFrame):
        seed(1)
        super(UserKNNAgeRecommender, self).__init__(data.urm_train)
        self.df_age = df_age.copy()
        self.data = data
        self._compute_item_score = self._compute_score_user_based

    def _compute_item_score_postprocess_for_cold_users(self, user_id_array, item_scores):
        return item_scores

    # WATCH OUT! HERE TOPK HAS A DIFFERENT MEANING
    # if topK == 0 -> take all person with the same age
    # if topK == 1 -> take all person with the same age or that differs by one
    def _compute_age_similarity(self, topK=0, shrink=100, distance_squared=True):
        # self.df_age.drop('data')
        list_u = self.df_age['row'].to_list()
        list_a = self.df_age['col'].to_list()
        age_dict = dict(zip(list_u, list_a))
        user_list = self.df_age['row'].tolist()
        user_ids = list(dict.fromkeys(user_list)) # no duplicates
        age_list_dict = {}
        data = []
        row_indices = []
        col_indices = []

        import time
        start = time.time()

        for i in range(1, 11):
            user_same_age = self.df_age['col'] == i
            user_same_age_df = self.df_age[user_same_age]
            user_same_age_list = user_same_age_df['row'].values.tolist()
            age_list_dict[i] = user_same_age_list

        for i in range(-8, 1):
            age_list_dict[i] = []
        for i in range(10, 18):
            age_list_dict[i] = []

        if topK == 0:
            for u_id in user_ids:
                u_age = age_dict[u_id]
                print("u id: " + str(u_id))
                user_same_age_list = age_list_dict[u_age]
                data += [1] * len(user_same_age_list)
                row_indices += [u_id] * len(user_same_age_list)
                col_indices += user_same_age_list
        elif topK == 1:
            for u_id in user_ids:
                u_age = age_dict[u_id]
                print("u id: " + str(u_id))
                user_same_age_list = age_list_dict[u_age]
                data += [1] * len(user_same_age_list)
                row_indices += [u_id] * len(user_same_age_list)
                col_indices += user_same_age_list
                if u_age != 10 and u_age != 1:
                    user_diff_age_list = age_list_dict[u_age + 1 or u_age - 1]
                elif u_age == 1:
                    user_diff_age_list = age_list_dict[u_age + 1]
                else:
                    user_diff_age_list = age_list_dict[u_age - 1]
                data += [0.2] * len(user_diff_age_list)
                row_indices += [u_id] * len(user_diff_age_list)
                col_indices += user_diff_age_list
        else:
            raise ValueError('TopK must be either 0 or 1 here.')

        n = self.data.number_of_users

        W_sparse = sps.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        print(W_sparse)
        print(W_sparse.shape)
        print(W_sparse.getnnz())
        stop = time.time()
        print("Similarity matrix created. Time passed: " + str(stop - start))
        return W_sparse



    def fit(self, topK=0, shrink=100, similarity='cosine', normalize=True, feature_weighting="none",
            **similarity_args):
        self.topK = topK
        self.shrink = shrink

        self.W_sparse = self._compute_age_similarity(topK=topK)
        self.W_sparse = check_matrix(self.W_sparse, format='csr')


