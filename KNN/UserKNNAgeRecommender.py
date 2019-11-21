#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender_utils import check_matrix
from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender

from Base.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np

from Base.Similarity.Compute_Similarity import Compute_Similarity
import pandas as pd
from random import seed
from random import random
import numpy as np
import scipy.sparse as sps


class UserKNNAgeRecommender(BaseSimilarityMatrixRecommender):
    """ UserKNN Content Based recommender
        It uses the age information given by the dataset.
    """


    RECOMMENDER_NAME = "UserKNNAgeRecommender"

    def __init__(self, df_age: pd.DataFrame, URM_train):
        seed(1)
        super(UserKNNAgeRecommender, self).__init__(URM_train)
        self.df_age = df_age.copy()
        self._compute_item_score = self._compute_score_user_based

    def _compute_item_score_postprocess_for_cold_users(self, user_id_array, item_scores):
        return item_scores

    def _compute_age_similarity(self, topK=50, shrink=100, distance_squared=True):


        user_list = self.df_age['row'].tolist()
        user_ids = list(dict.fromkeys(user_list)) # no duplicates
        number_of_users = len(user_ids)
        max_id = max(user_ids)
        knn = min(topK, number_of_users)
        s = np.zeros((max_id +1, max_id+1))
        # W_dense = np.zeros((max_id +1, max_id+1))
        data = []
        row_indices = []
        col_indices = []



        for i in range(0,number_of_users):
            import time
            start = time.time()

            u_id = user_ids[i]
            u_age = self._get_age(u_id)
            s[u_id] = np.zeros(max_id+1)
            for j in range(i+1, number_of_users):
                v_id = user_ids[j]
                v_age = self._get_age(v_id)
                s[u_id][v_id] = self._simil(u_age, v_age)

            # for symmetry
            if u_id!=0:
                for k in range(0, u_id):
                    s[u_id][k] = s[u_id-1][k]

            indices_topK = np.argpartition(s[u_id], -knn)[-knn:]
            for _ in range(0,knn):
                row_indices.append([u_id])


            #print("indices_topK")
            #print(indices_topK)
            #print(u_id)
            for idx in indices_topK:
                data.append(s[u_id][idx])

            col_indices.append(list(indices_topK))

            if(len(row_indices) == len(data) == len(col_indices) == knn):
                print("correct dimensions")
            else:
                print("wrong dimensions")
            stop = time.time()
            print("time passed: " + str(stop-start))

        W_sparse = sps.csr_matrix(data, (row_indices, col_indices))
        return W_sparse





    def _simil(self, age_1, age_2):
        val = 1 - (abs(age_1 - age_2)/7)
        return val

    def _get_age(self, user_id):
        user_rows = self.df_age['row'] == user_id
        ages_list = self.df_age[user_rows].drop(columns='row')['col'].tolist()
        length_ages_list = len(ages_list)
        if(length_ages_list==1):
            return ages_list.pop()
        else:
            index = int(random() * (length_ages_list - 1))
            return ages_list.pop(index)




    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting="none",
            **similarity_args):
        self.topK = topK
        self.shrink = shrink

        self.W_sparse = self._compute_age_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')