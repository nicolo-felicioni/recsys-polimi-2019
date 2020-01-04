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
from Hybrid.Hybrid400AlphaRecommender import Hybrid400AlphaRecommender
from Hybrid.Score.Hybrid3ScoreSubRecommender import Hybrid3ScoreSubRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


class Hybrid600AlphaRecommender(BaseRecommender):
    """Hybrid600AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid600AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid600AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.cold_recommender = Hybrid100AlphaRecommender(data)
        self.warm_1_recommender = Hybrid101AlphaRecommender(data)

        self.warm_recommender = Hybrid3ScoreSubRecommender(data=data)


    def fit(self):
        self.cold_recommender.fit()
        self.warm_1_recommender.fit()
        self.warm_recommender.fit(alpha_beta_ratio=0.5, alpha_gamma_ratio=0.3)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        if user_id_array in self.data.urm_train_users_by_type[0][1]:
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
