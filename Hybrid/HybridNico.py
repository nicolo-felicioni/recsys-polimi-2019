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
        self.data = data
        self.random_seed = random_seed
        self.slim = SLIMElasticNetRecommender(data.urm_train)
        self.rp3 = RP3betaRecommender(data.urm_train)


    def fit(self, alpha=0.5):
        self.slim.load_model('', 'SLIM_ElasticNetURM_seed='
                             + str(self.random_seed) +
                             '_topK=100_l1_ratio=0.02_alpha=0.0005_positive_only=True_max_iter=35')
        self.rp3.fit(topK=20, alpha=0.16, beta=0.24)


        path_slim = 'slim_item_scores_random_seed=' + str(self.random_seed)
        path_rp3 = 'slim_item_scores_random_seed=' + str(self.random_seed)

        # cache = (os.path.exists(path_slim + '.npy') and os.path.exists(path_rp3 + '.npy'))
        cache = False

        print("cache is")
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

        n_users = self.data.number_of_users
        n_item = self.data.number_of_items
        list_slim = []
        list_rp3 = []
        for u_id in range(n_users):
            # if u_id % 1000 == 0:
            #     print('user: {} / {}'.format(u_id, n_users - 1))
            list_slim.append(np.squeeze(self.slim._compute_item_score(user_id_array=u_id)))
            list_rp3.append(np.squeeze(self.rp3._compute_item_score(user_id_array=u_id)))

        mat_scores_slim = np.stack(list_slim, axis=0)
        mat_scores_rp3 = np.stack(list_rp3, axis=0)

        '''print("slim scores stats:")
        print("min = {}".format(mat_scores_slim.min()))
        print("max = {}".format(mat_scores_slim.max()))
        print("average = {}".format(mat_scores_slim.mean()))


        print("rp3 scores stats:")
        print("min = {}".format(mat_scores_rp3.min()))
        print("max = {}".format(mat_scores_rp3.max()))
        print("average = {}".format(mat_scores_rp3.mean()))'''

        # normalization
        mat_scores_slim /= mat_scores_slim.max()
        mat_scores_rp3 /= mat_scores_rp3.max()

        self.mat_scores_slim = mat_scores_slim
        self.mat_scores_rp3 = mat_scores_rp3

        '''print("slim scores stats:")
        print("min = {}".format(mat_scores_slim.min()))
        print("max = {}".format(mat_scores_slim.max()))
        print("average = {}".format(mat_scores_slim.mean()))

        print("rp3 scores stats:")
        print("min = {}".format(mat_scores_rp3.min()))
        print("max = {}".format(mat_scores_rp3.max()))
        print("average = {}".format(mat_scores_rp3.mean()))'''

        # np.save(path_slim, arr=mat_scores_slim)
        # np.save(path_rp3, arr=mat_scores_rp3)

        self.score_matrix = alpha * mat_scores_slim + (1 - alpha) * mat_scores_rp3





    def _compute_item_score(self, user_id_array, items_to_compute = None):
        return self.score_matrix[user_id_array]


