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
from Hybrid.Hybrid300AlphaRecommender import Hybrid300AlphaRecommender
from KNN.ItemKNNCBFOnlyColdRecommender import ItemKNNCBFOnlyColdRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


class Hybrid004AlphaRecommender(BaseRecommender):
    """Hybrid004AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid004AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid004AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.warm_recommender = Hybrid300AlphaRecommender(data)
        self.warm_1_recommender = Hybrid101AlphaRecommender(data)
        # self.warm_2_recommender = Hybrid102AlphaRecommender(data)
        self.warm_7_recommender = Hybrid300AlphaRecommender(data)
        self.cold_recommender = Hybrid100AlphaRecommender(data)

    def fit(self):
        self.cold_recommender.fit()
        rec_augmenter_cold_item = ItemKNNCBFOnlyColdRecommender(self.data)
        # r2 = ItemKNNCBFRecommender(data.urm_train, data.icm_all_augmented)
        rec_augmenter_cold_item.fit(topK=10, shrink=1, feature_weighting="BM25")
        self.warm_recommender.augment(rec_augmenter_cold_item,
                                      self.data.ids_warm_train_users,
                                      self.warm_recommender.augmented_matrix, cutoff=3, amount=1)
        rec_warm = RP3betaRecommender(self.warm_recommender.augmented_matrix)
        rec_warm.fit(topK=20, alpha=0.16, beta=0.24)
        self.warm_recommender.fit(rec_warm)
        self.warm_1_recommender.fit()
        self.warm_7_recommender.augment(rec_augmenter_cold_item,
                                        self.data.ids_warm_train_users,
                                        self.warm_7_recommender.augmented_matrix, cutoff=3, amount=1)
        rec_warm_7 = ItemKNNCFRecommender(self.warm_7_recommender.augmented_matrix)
        rec_warm_7.fit(topK=22, shrink=850, feature_weighting="BM25", similarity="jaccard")
        self.warm_7_recommender.fit(rec_warm_7)

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
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.urm_train_users_by_type[10][1]:
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.urm_train_users_by_type[11][1]:
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        elif user_id_array in self.data.ids_cold_user:
            return self.cold_recommender.recommend(user_id_array, cutoff=cutoff)
