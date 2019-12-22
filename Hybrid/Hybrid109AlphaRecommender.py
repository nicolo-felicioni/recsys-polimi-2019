from Base.BaseRecommender import BaseRecommender
from Base.NonPersonalizedRecommender import TopPop
from DataObject import DataObject
from Data_manager.DataReader import DataReader
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from Hybrid.Hybrid1CXAlphaRecommender import Hybrid1CXAlphaRecommender
from Hybrid.Hybrid1XXAlphaRecommender import Hybrid1XXAlphaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_AsySVD_Cython
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import numpy as np
import operator


class Hybrid109AlphaRecommender(BaseRecommender):
    """Hybrid109AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid109AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid109AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec1 = RP3betaRecommender(data.urm_train)
        self.rec2 = ItemKNNCFRecommender(data.urm_train)
        self.rec1.fit(topK=20, alpha=0.11, beta=0.18)
        self.rec2.fit(topK=18, shrink=850, similarity='jaccard', feature_weighting='BM25')
        cold = data.ids_cold_user
        self.target_users = data.urm_train_users_by_type[10][1]
        self.hybrid_rec = Hybrid1CXAlphaRecommender(data, recommenders=[self.rec1, self.rec2],
                                                    recommended_users=self.target_users, max_cutoff=30)

    def fit(self):
        weights = [
            [1.0, 0.7342122774376785, 0.34497645278822586, 0.36927385971216775, 0.2656415434217909, 0.1919421372364918,
             0.13293725674498652, 0.14834334445233766, 0.09775023936714271, 0.10937034954552564, 0.12423098623036653,
             0.12128788329523746, 0.09922803890669518, 0.11319974533876793, 0.12225324687713784, 0.12652916395766228,
             0.1455085385513116, 0.0914697268325847, 0.0864212005177428, 0.0927393080791469, 0.07406451741135928,
             0.07317443382010559, 0.05931225834380707, 0.0648359502039145, 0.048739304754641055, 0.03240929388889551,
             0.03666804622590347, 0.04216825315978899, 0.0447766477977417, 0.0],
            [0.4408648438266093, 0.4922839735436428, 0.24278674016874366, 0.2625681515299236, 0.20164121702221108,
             0.1910052083663528, 0.16326625410923354, 0.0923349547002018, 0.0945701088881237, 0.10221568403081797,
             0.11754803663544067, 0.1301553725839876, 0.11021196426884693, 0.11344510775804244, 0.11457061547288103,
             0.08422114145406502, 0.09685431267217476, 0.10379217500012262, 0.10710455074654938, 0.07314029501013618,
             0.08298821173426496, 0.09228327804768503, 0.07716595603811922, 0.060006078819213785, 0.06542915168076886,
             0.05415221757898212, 0.03694072713955989, 0.028810312392733607, 0.03269610716181971, 0.0]]
        self.hybrid_rec.weights = weights

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        if user_id_array in self.target_users:
            return self.hybrid_rec.recommend(user_id_array=user_id_array, cutoff=cutoff)
        else:
            return []
