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
import os

from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

class HybridNico(BaseRecommender):
    """Hybrid100AlphaRecommender recommender"""

    RECOMMENDER_NAME = "HybridNico"

    def __init__(self, data: DataObject, random_seed: int):
        super(HybridNico, self).__init__(data.urm_train)
        self.random_seed = random_seed
        self.slim = SLIMElasticNetRecommender(self.URM_train)
        self.rp3 = RP3betaRecommender(self.URM_train)
        self.number_of_users = data.number_of_users


    def fit(self, alpha=0.5):
        self.slim.load_model('', 'SLIM_ElasticNetURM_seed='
                             + str(self.random_seed) +
                             '_topK=100_l1_ratio=0.04705_alpha=0.00115_positive_only=True_max_iter=35')
        self.rp3.fit(topK=20, alpha=0.16, beta=0.24)


        path_slim = 'slim_item_scores_random_seed=' + str(self.random_seed)
        path_rp3 = 'slim_item_scores_random_seed=' + str(self.random_seed)

        # cache = (os.path.exists(path_slim + '.npy') and os.path.exists(path_rp3 + '.npy'))
        cache = False

        print("cache is hardcoded to:")
        print(cache)

        if cache:
            self.fit_cached(path_slim=path_slim, path_rp3=path_rp3, alpha=alpha)
        else:
            self.fit_no_cached(path_slim=path_slim, path_rp3=path_rp3, alpha=alpha)


    def fit_cached(self, path_slim, path_rp3, alpha=0.5):
        mat_scores_slim = np.load(path_slim + '.npy')
        mat_scores_rp3 = np.load(path_rp3 + '.npy')

        self.score_matrix = alpha * mat_scores_slim + (1 - alpha) * mat_scores_rp3

    def fit_no_cached(self, path_slim, path_rp3, alpha=0.5):

        user_id_array = np.array(range(self.number_of_users))

        self.mat_scores_slim = self.slim._compute_item_score(user_id_array=user_id_array)
        self.mat_scores_rp3 = self.rp3._compute_item_score(user_id_array=user_id_array)

        # normalization
        self.mat_scores_slim /= self.mat_scores_slim.max()
        self.mat_scores_rp3 /= self.mat_scores_rp3.max()

        # np.save(path_slim, arr=mat_scores_slim)
        # np.save(path_rp3, arr=mat_scores_rp3)

        self.score_matrix = alpha * self.mat_scores_slim + (1 - alpha) * self.mat_scores_rp3





    def _compute_item_score(self, user_id_array, items_to_compute = None):
        return self.score_matrix[user_id_array]


