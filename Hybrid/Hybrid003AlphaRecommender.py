from Base.BaseRecommender import BaseRecommender
from DataObject import DataObject
from Data_manager.DataReader import DataReader
from GraphBased.P3alphaRecommender import P3alphaRecommender
from Hybrid.Hybrid100AlphaRecommender import Hybrid100AlphaRecommender
from Hybrid.Hybrid101AlphaRecommender import Hybrid101AlphaRecommender
from Hybrid.Hybrid1XXAlphaRecommender import Hybrid1XXAlphaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


class Hybrid003AlphaRecommender(BaseRecommender):
    """Hybrid003AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid003AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid003AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.poco_warm_recommender = Hybrid100AlphaRecommender(data)
        self.quasi_warm_recommender = Hybrid101AlphaRecommender(data)
        # quasi quasi
        rec1 = ItemKNNCFRecommender(self.data.urm_train)
        rec1.fit(shrink=15, topK=12, feature_weighting="none")
        rec2 = P3alphaRecommender(self.data.urm_train)
        rec2.fit(topK=170, implicit=True, alpha=0.5)
        quasi_quasi_base_recommenders = [rec1, rec2]
        weights = []
        self.quasi_quasi_warm_recommender = Hybrid1XXAlphaRecommender(data, quasi_quasi_base_recommenders)
        self.warm_recommender = ItemKNNCFRecommender(data.urm_train)
        self.tanto_warm_recommender = ItemKNNCFRecommender(data.urm_train)
        self.cold_recommender = UserKNNCBFRecommender(data.ucm_all, data.urm_train)

    def fit(self):
        self.poco_warm_recommender.fit()
        # self.poco_warm_recommender.fit(topK=20000, shrink=20000, feature_weighting="TF-IDF")
        self.quasi_warm_recommender.fit()
        self.quasi_quasi_warm_recommender
        self.warm_recommender.fit(topK=12, shrink=15, feature_weighting="none")
        self.tanto_warm_recommender.fit(topK=12, shrink=15, feature_weighting="none", similarity="jaccard")
        self.cold_recommender.fit(topK=11000, shrink=1)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        if user_id_array in self.data.urm_train_users_by_type[0][1]:
            return self.cold_recommender.recommend(user_id_array, cutoff=cutoff)
        if user_id_array in self.data.urm_train_users_by_type[1][1]:
            return self.poco_warm_recommender.recommend(user_id_array, cutoff=cutoff)
        if user_id_array in self.data.urm_train_users_by_type[2][1]:
            return self.quasi_warm_recommender.recommend(user_id_array, cutoff=cutoff)
        if user_id_array in self.data.urm_train_users_by_type[3][1]:
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        if user_id_array in self.data.urm_train_users_by_type[4][1]:
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        if user_id_array in self.data.urm_train_users_by_type[5][1]:
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        if user_id_array in self.data.urm_train_users_by_type[6][1]:
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        if user_id_array in self.data.urm_train_users_by_type[7][1]:
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        if user_id_array in self.data.urm_train_users_by_type[8][1]:
            return self.tanto_warm_recommender.recommend(user_id_array, cutoff=cutoff)
        if user_id_array in self.data.urm_train_users_by_type[9][1]:
            return self.tanto_warm_recommender.recommend(user_id_array, cutoff=cutoff)