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
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
import scipy.sparse as sps


class Hybrid005AlphaRecommender(BaseRecommender):
    """Hybrid004AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid005AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid005AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.cold_recommender = Hybrid100AlphaRecommender(data)
        self.cold_recommender.fit()
        # rec1 = RP3betaRecommender(data.urm_train)
        # rec1.fit(topK=20, alpha=0.12, beta=0.24)
        # rec2 = ItemKNNCFRecommender(data.urm_train)
        # rec2.fit(topK=22, shrink=850, similarity='jaccard', feature_weighting='BM25')
        # self.warm_2_recommender = ItemKNNSimilarityHybridRecommender(data.urm_train, rec1.W_sparse, rec2.W_sparse)
        urm = data.urm_train
        urm = sps.vstack([data.urm_train, data.icm_all_augmented.T])
        urm = urm.tocsr()
        self.warm_recommender = MultiThreadSLIM_ElasticNet(data.urm_train)
        self.warm_2_3_recommender = MultiThreadSLIM_ElasticNet(data.urm_train)
        self.warm_1_recommender = Hybrid101AlphaRecommender(data)

    def fit(self):
        # self.warm_recommender.fit(topK=30, shrink=30, feature_weighting="none", similarity="jaccard")
        # self.warm_2_recommender.fit(alpha=0.9, topK=50)
        self.warm_1_recommender.fit()
        try:
            self.warm_recommender.load_model("SLIM_ElasticNet_ICM",
                                             "FULL_URM_topK=100_l1_ratio=0.04705_alpha=0.00115_positive_only=True_max_iter=35")
        except:
            self.warm_recommender.fit(topK=100, l1_ratio=0.04705, alpha=0.00115, positive_only=True, max_iter=35)
            self.warm_recommender.save_model("SLIM_ElasticNet_ICM",
                                             "FULL_URM_topK=100_l1_ratio=0.04705_alpha=0.00115_positive_only=True_max_iter=35")

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
