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

class HybridScoreRecommender(BaseRecommender):
    """HybridScoreRecommender recommender"""

    RECOMMENDER_NAME = "HybridScoreRecommender"

    def __init__(self, data: DataObject, random_seed: int):
        super(HybridScoreRecommender, self).__init__(data.urm_train)
        self.random_seed = random_seed
        self.slim = SLIMElasticNetRecommender(self.URM_train)
        self.rp3 = RP3betaRecommender(self.URM_train)


    def fit(self, alpha=0.5):
        self.slim.load_model('', 'SLIM_ElasticNetURM_seed='
                             + str(self.random_seed) +
                             '_topK=100_l1_ratio=0.04705_alpha=0.00115_positive_only=True_max_iter=35')
        self.rp3.fit(topK=20, alpha=0.16, beta=0.24)
        self.alpha = alpha


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        # ATTENTION!
        # THIS METHOD WORKS ONLY IF user_id_array IS A SCALAR AND NOT AN ARRAY
        # TODO

        scores_slim = self.slim._compute_item_score(user_id_array=user_id_array)
        scores_rp3 = self.rp3._compute_item_score(user_id_array=user_id_array)

        # normalization
        slim_max = scores_slim.max()
        rp3_max = scores_rp3.max()

        if not slim_max == 0:
            scores_slim /= slim_max
        if not rp3_max == 0:
            scores_rp3 /= rp3_max

        scores_total = self.alpha * scores_slim + (1 - self.alpha) * scores_rp3
        return scores_total

