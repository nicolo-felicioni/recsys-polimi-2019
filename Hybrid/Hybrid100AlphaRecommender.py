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


class Hybrid100AlphaRecommender(BaseRecommender):
    """Hybrid100AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid100AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid100AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec1 = UserKNNCBFRecommender(data.ucm_all, data.urm_train)
        self.rec2 = TopPop(data.urm_train)
        self.rec1.fit(shrink=1, topK=11000)
        self.rec2.fit()
        self.hybrid_rec = Hybrid1XXAlphaRecommender(data, recommenders=[self.rec1, self.rec2], max_cutoff=30)

    def fit(self):
        weights = [[19.53, 11.84, 14.84, 12.36, 12.56, 12.07, 7.57, 6.79, 7.47, 7.12, 7.74, 5.74,
                    5.62, 5.99, 7.04, 7.21, 7.58, 8.72, 9.63, 9.29, 8.82, 8.5, 8.29, 8.28,
                    9.65, 9.17, 9.76, 8.32, 7.03, 6.96],
                   [11.02, 12.8, 10.18, 7.3, 7.65, 8.82, 8.68, 8.39, 4.88, 11.28, 11.68, 11.02,
                    9.61, 9.07, 7.89, 6.83, 4.84, 3.11, 4.78, 4.37, 5.63, 6.84, 8.22, 6.08,
                    5.8, 6.83, 6.37, 3.64, 4.04, 6.]]
        # weights = [[20.75, 12., 15.25, 12.25, 9.75, 12.25, 8.5, 5., 8.5,
        #             9.75, 8.25, 5., 4.75, 7.25, 6.75, 6.25, 8., 8.,
        #             9.5, 7.25, 8.75, 8.25, 6.75, 10.25, 12.25, 8., 8.5,
        #             10., 7.75, 6.5],
        #            [11., 13., 10., 5.25, 7.5, 10., 9.25, 9., 4.,
        #             12.5, 10.75, 11.75, 14., 8.5, 6.5, 6.5, 3.75, 1.75,
        #             6.25, 7.25, 6.25, 8.5, 1.75, 7., 5.5, 6.25, 5.5,
        #             4.25, 3., 4.5]]
        self.hybrid_rec.fit(weights=weights)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        return self.hybrid_rec.recommend(user_id_array=user_id_array, cutoff=cutoff)
