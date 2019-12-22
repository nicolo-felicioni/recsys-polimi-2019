from Base.BaseRecommender import BaseRecommender
from Base.NonPersonalizedRecommender import TopPop
from DataObject import DataObject
from Data_manager.DataReader import DataReader
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from Hybrid.Hybrid003AlphaRecommender import Hybrid003AlphaRecommender
from Hybrid.Hybrid1CXAlphaRecommender import Hybrid1CXAlphaRecommender
from Hybrid.Hybrid1XXAlphaRecommender import Hybrid1XXAlphaRecommender
from KNN.ItemKNNCBFOnlyColdRecommender import ItemKNNCBFOnlyColdRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import numpy as np
import operator


class Hybrid300AlphaRecommender(BaseRecommender):
    """Hybrid300AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid300AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid300AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.augmented_matrix = data.urm_train

    def augment(self, rec, users, data, cutoff=10, amount=1):
        matrix = data.tolil(copy=True)
        for user in users:
            items = rec.recommend(user, cutoff=cutoff)
            for item in items:
                matrix[user, item] += amount
        self.augmented_matrix = matrix.tocsr()

    def reduce(self, rec, users, data, cutoff=10, amount=1):
        matrix = data.tolil(copy=True)
        for user in users:
            items = rec.recommend(user, cutoff=cutoff, remove_seen_flag=False)
            not_relevant_items = [x
                                  for x in matrix[user].rows[0]
                                  if x not in items]
            for nr_item in not_relevant_items:
                if matrix[user, nr_item] > 0:
                    matrix[user, nr_item] -= amount
        self.augmented_matrix = matrix.tocsr()


        self.augmented_matrix = matrix.tocsr()


    def fit(self, rec):
        self.rec = rec

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        new_cutoff = len(self.data.urm_train[user_id_array].indices) + cutoff
        recommended_items = self.rec.recommend(user_id_array=user_id_array, cutoff=new_cutoff, remove_seen_flag=False)
        recommended_items_not_seen = [x
                                      for x in recommended_items
                                      if x not in self.data.urm_train[user_id_array].indices]
        return recommended_items_not_seen[:cutoff]
