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
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender


class ItemKNNCBFOnlyColdRecommender(BaseSimilarityMatrixRecommender):

    def __init__(self, data):
        super(ItemKNNCBFOnlyColdRecommender, self).__init__(data.urm_train)
        self.rec = ItemKNNCBFRecommender(data.urm_train, data.icm_all_augmented)
        self.cold_item = np.concatenate((data.ids_cold_train_items, data.ids_cold_item))
        self.data = data

    def _compute_item_score_postprocess_for_cold_items(self, item_scores):
        """
        In CBF no cold items are to be removed
        :param item_scores:
        :return:
        """

        return item_scores

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting="none",
            **similarity_args):
        self.rec.fit(topK=topK, shrink=shrink, similarity=similarity, normalize=normalize,
                     feature_weighting=feature_weighting, **similarity_args)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        recommended_items = self.rec.recommend(user_id_array=user_id_array, cutoff=cutoff, remove_seen_flag=False)
        recommended_items = np.array([x for x in recommended_items if
                                      x in self.cold_item and x not in self.data.urm_train[user_id_array].indices])
        recommended_items = np.pad(recommended_items, (0, cutoff - recommended_items.shape[0]), 'constant', constant_values=-1)
        return recommended_items

    def recommend_no_pad(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                         remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        recommended_items = self.rec.recommend(user_id_array=user_id_array, cutoff=cutoff, remove_seen_flag=False)
        recommended_items = np.array([x for x in recommended_items if
                                      x in self.cold_item and x not in self.data.urm_train[user_id_array].indices])
        return recommended_items
