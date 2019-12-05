from Base.BaseRecommender import BaseRecommender
from DataObject import DataObject
from Data_manager.DataReader import DataReader
from GraphBased.P3alphaRecommender import P3alphaRecommender
from Hybrid.Hybrid100AlphaRecommender import Hybrid100AlphaRecommender
from Hybrid.Hybrid101AlphaRecommender import Hybrid101AlphaRecommender
from Hybrid.Hybrid102AlphaRecommender import Hybrid102AlphaRecommender
from Hybrid.Hybrid105AlphaRecommender import Hybrid105AlphaRecommender
from Hybrid.Hybrid108AlphaRecommender import Hybrid108AlphaRecommender
from Hybrid.Hybrid109AlphaRecommender import Hybrid109AlphaRecommender
from Hybrid.Hybrid1XXAlphaRecommender import Hybrid1XXAlphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


class Hybrid004AlphaRecommender(BaseRecommender):
    """Hybrid004AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid004AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid004AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.warm_recommender = RP3betaRecommender(data.urm_train)
        self.warm_1_recommender = Hybrid101AlphaRecommender(data)
        self.warm_11_recommender = Hybrid109AlphaRecommender(data)
        self.cold_recommender = Hybrid100AlphaRecommender(data)

    def fit(self):
        self.cold_recommender.fit()
        # self.warm_recommender.fit(topK=30, shrink=30, feature_weighting="none", similarity="jaccard")
        self.warm_recommender.fit(topK=30, alpha=0.24, beta=0.24)
        self.warm_1_recommender.fit()
        self.warm_11_recommender.fit()

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        if user_id_array in self.data.ids_ultra_cold_users:
            return self.cold_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.urm_train_users_by_type[0][1]:
            return self.cold_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.urm_train_users_by_type[1][1]:
            return self.warm_1_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.urm_train_users_by_type[2][1]:
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.urm_train_users_by_type[3][1]:
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.urm_train_users_by_type[4][1]:
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.urm_train_users_by_type[5][1]:
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.urm_train_users_by_type[6][1]:
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.urm_train_users_by_type[7][1]:
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.urm_train_users_by_type[8][1]:
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.urm_train_users_by_type[9][1]:
            return self.warm_11_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.urm_train_users_by_type[10][1]:
            return self.warm_11_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.urm_train_users_by_type[11][1]:
            return self.warm_11_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.ids_cold_user:
            return self.cold_recommender.recommend(user_id_array, cutoff=cutoff)
