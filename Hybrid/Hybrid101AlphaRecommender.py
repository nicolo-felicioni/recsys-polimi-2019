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


class Hybrid101AlphaRecommender(BaseRecommender):
    """Hybrid101AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid101AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid101AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec1 = SLIM_BPR_Cython(data.urm_train)
        self.rec2 = ItemKNNCFRecommender(data.urm_train)
        self.rec3 = RP3betaRecommender(data.urm_train)
        self.rec1.fit(sgd_mode="adagrad", topK=15000, epochs=250, learning_rate=1e-05, lambda_i=0.01, lambda_j=0.01)
        # self.rec1.fit(topK=435, epochs=115, symmetric=True, sgd_mode='adagrad', lambda_i=0.0010067865845523253,
        #     lambda_j=3.764224446055186e-05, learning_rate=0.00024125117955549121)
        self.rec2.fit(topK=20000, shrink=20000, feature_weighting="TF-IDF")
        self.rec3.fit(topK=10000, alpha=0.55, beta=0.01, implicit=True, normalize_similarity=True)
        self.hybrid_rec = Hybrid1XXAlphaRecommender(data, recommenders=[self.rec1, self.rec2, self.rec3], max_cutoff=20)

    def fit(self):
        weights = [[69.4, 25.7, 11.7, 9.4, 8.4, 5.4, 6.6, 6., 5.5, 5.6, 5., 4.4, 3.3, 5.7,
                    4.2, 3.7, 4.5, 2.8, 3.8, 3.4],
                   [77.8, 29.3, 17.4, 9., 8.5, 8.9, 5.9, 5.9, 5.4, 5.1, 6., 6.3, 4.4, 4.6,
                    5.2, 4.9, 3.5, 3.3, 3.5, 4.3],
                   [78.5, 29.2, 15.6, 10.9, 9.4, 6.5, 8.3, 5.7, 6.3, 6.6, 4.3, 4.2, 4.3, 4.6,
                    6.1, 4.7, 5.1, 4.7, 4.9, 5.1]]
        self.hybrid_rec.fit(weights=weights)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        return self.hybrid_rec.recommend(user_id_array=user_id_array, cutoff=cutoff)
