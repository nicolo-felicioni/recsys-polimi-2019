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
import scipy.sparse as sps


class Hybrid201AlphaRecommender(BaseRecommender):
    """Hybrid201AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid201AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid201AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        urm = data.urm_train
        urm = sps.vstack([data.urm_train, data.icm_all_augmented.T])
        urm = urm.tocsr()
        self.rec1 = SLIM_BPR_Cython(urm)
        self.rec2 = ItemKNNCFRecommender(urm)
        self.rec3 = RP3betaRecommender(urm)
        self.random_seed = data.random_seed
        try:
            self.rec1.load_model("stored_recommenders/slim_bpr/",
                                 f'with_icm_{self.random_seed}_topK=15000_epochs=250_learning_rate=1e-05_lambda_i=0.01_lambda_j=0.01')
        except:
            self.rec1.fit(sgd_mode="adagrad", topK=15000, epochs=250, learning_rate=1e-05, lambda_i=0.01, lambda_j=0.01)
            self.rec1.save_model("stored_recommenders/slim_bpr/",
                                 f'with_icm_{self.random_seed}_topK=15000_epochs=250_learning_rate=1e-05_lambda_i=0.01_lambda_j=0.01')
        try:
            self.rec2.load_model("stored_recommenders/item_cf/",
                                 f'with_icm_{self.random_seed}_topK=20000_shrink=20000_feature_weighting=TF-IDF')
        except:
            self.rec2.fit(topK=20000, shrink=20000, feature_weighting="TF-IDF")
            self.rec2.save_model("stored_recommenders/item_cf/",
                                 f'with_icm_{self.random_seed}_topK=20000_shrink=20000_feature_weighting=TF-IDF')
        try:
            self.rec3.load_model("stored_recommenders/rp3_beta/",
                                 f'with_icm_{self.random_seed}_topK=10000_alpha=0.55_beta=0.01_implicit=True_normalize_similarity=True')
        except:
            self.rec3.fit(topK=10000, alpha=0.55, beta=0.01, implicit=True, normalize_similarity=True)
            self.rec3.save_model("stored_recommenders/rp3_beta/",
                                 f'with_icm_{self.random_seed}_topK=10000_alpha=0.55_beta=0.01_implicit=True_normalize_similarity=True')
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
