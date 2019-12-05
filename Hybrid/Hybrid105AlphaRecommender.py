from Base.BaseRecommender import BaseRecommender
from Base.NonPersonalizedRecommender import TopPop
from DataObject import DataObject
from Data_manager.DataReader import DataReader
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from Hybrid.Hybrid1XXAlphaRecommender import Hybrid1XXAlphaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import numpy as np
import operator


class Hybrid105AlphaRecommender(BaseRecommender):
    """Hybrid105AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid105AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid105AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        # self.rec1 = SLIM_BPR_Cython(data.urm_train)
        self.rec2 = ItemKNNCFRecommender(data.urm_train)
        self.rec3 = RP3betaRecommender(data.urm_train)
        # self.rec1.fit(sgd_mode="adagrad", topK=50, epochs=150, learning_rate=1e-05, lambda_i=1, lambda_j=0.001)
        self.rec2.fit(topK=10, shrink=30, similarity="tanimoto")
        self.rec3.fit(topK=30, alpha=0.24, beta=0.24, implicit=True, normalize_similarity=True)

        self.hybrid_rec = Hybrid1XXAlphaRecommender(data, recommenders=[self.rec2, self.rec3], max_cutoff=20)

    def fit(self):
        weights = np.array([[91.45, 46.9, 36.6, 26.75, 19.85, 18.45, 14.95, 11.9, 14.25, 9.9, 10.65, 10.65,
                             9.85, 9.2, 7.6, 7.8, 6.6, 7.05, 5.8, 6.25, 4.7, 5., 5.45, 4.45,
                             3.8, 4.05, 4.05, 4.05, 3.75, 3.95],
                            [94.7, 51.2, 34.25, 26.45, 22.25, 17.55, 15.7, 13.7, 10.75, 9.9, 9.15, 8.2,
                             6.6, 6.35, 6.65, 5.45, 5.25, 4.4, 5.6, 3.95, 3.2, 3.85, 2.6, 2.85,
                             2.75, 2.3, 2.55, 2.4, 2.9, 1.6]])
        weights[0] = weights[0] / 3 * 2
        self.hybrid_rec.fit(weights=weights)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        return self.hybrid_rec.recommend(user_id_array=user_id_array, cutoff=cutoff)

# class Hybrid105AlphaRecommender(BaseRecommender):
#     """Hybrid105AlphaRecommender recommender"""
#
#     RECOMMENDER_NAME = "Hybrid105AlphaRecommender"
#
#     def __init__(self, data: DataObject):
#         super(Hybrid105AlphaRecommender, self).__init__(data.urm_train)
#         self.data = data
#         self.rec1 = SLIM_BPR_Cython(data.urm_train)
#         self.rec2 = ItemKNNCFRecommender(data.urm_train)
#         self.rec3 = RP3betaRecommender(data.urm_train)
#         self.rec1.fit(sgd_mode="adagrad", topK=50, epochs=150, learning_rate=1e-05, lambda_i=1, lambda_j=0.001)
#         self.rec2.fit(topK=30, shrink=30, similarity="tanimoto")
#         self.rec3.fit(topK=20, alpha=0.16, beta=0.24, implicit=True, normalize_similarity=True)
#
#         self.hybrid_rec = Hybrid1XXAlphaRecommender(data, recommenders=[self.rec1, self.rec2, self.rec3], max_cutoff=12)
#
#     def fit(self):
#         weights = [[716.1, 354.4, 238.2, 182.4, 141.7, 122.9, 107.3, 99.1, 95.9, 84.1, 76.8, 73.9,
#                     68.9, 68.1, 62.6, 62.6, 57.8, 58.9, 52.7, 54.],
#                    [820.5, 388.9, 279.9, 208.5, 160.7, 137.6, 112.6, 110.7, 94.9, 87.7, 80.3, 78.8,
#                     72.7, 66.7, 65.3, 64.1, 58.6, 56.7, 52.4, 50.5],
#                    [813.5, 396.9, 272.1, 210.7, 174.1, 136.9, 115.3, 93.4, 83.7, 76.4, 70.9, 64.,
#                     58.6, 52.4, 48.2, 46.4, 40.8, 42.8, 36.6, 35.1]]
#         self.hybrid_rec.fit(weights=weights)
#
#     def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
#                   remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
#         return self.hybrid_rec.recommend(user_id_array=user_id_array, cutoff=cutoff)
