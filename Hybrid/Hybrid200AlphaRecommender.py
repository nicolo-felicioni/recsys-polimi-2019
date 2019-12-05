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


class Hybrid200AlphaRecommender(BaseRecommender):
    """Hybrid200AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid200AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid200AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec = Hybrid003AlphaRecommender(data)
        self.cold_item_rec = ItemKNNCBFOnlyColdRecommender(data)
        self.cold_item_rec.fit(topK=10, shrink=20)

    def fit(self):
        self.rec.fit()

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        recommended_cold_item = self.cold_item_rec.recommend_no_pad(user_id_array=user_id_array, cutoff=4)
        recommended_item = self.rec.recommend(user_id_array=user_id_array, cutoff=cutoff)
        if recommended_cold_item.shape[0] > 0:
            recommended_result = np.concatenate((recommended_item[:-recommended_cold_item.shape[0]], recommended_cold_item))
            return recommended_result
        else:
            return recommended_item
