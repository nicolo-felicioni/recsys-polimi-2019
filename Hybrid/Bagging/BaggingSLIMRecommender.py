import copy

from joblib import Parallel, delayed

from Base.BaseRecommender import BaseRecommender
from Base.NonPersonalizedRecommender import TopPop
from DataObject import DataObject, augment_with_item_similarity_best_scores, augment_with_user_similarity_best_scores, \
    augment_with_best_recommended_items
from Data_manager.AdvancedSplitter import split, split_with_triple
from GraphBased.RP3betaRecommender import RP3betaRecommender
from Hybrid.Score.HybridItemSimilarityRecommender import HybridItemSimilarityRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import numpy as np
import scipy.sparse as sps

from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender


class par:
    def __init__(self,
                 data: DataObject,
                 split_policy=None,
                 topK=100,
                 l1_ratio=0.04705,
                 alpha=0.00115,
                 positive_only=True,
                 max_iter=35,
                 ):

        self.data = data
        self.split_policy = split_policy
        self.topK = topK
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.max_iter = max_iter

        # Initialize the split policy if not already done
        if split_policy is None:
            self.split_policy = np.array([
                [(0, 0), 0, 0.0],
                [(1, 1), 0, 0.0],
                [(2, 2), 0, 0.05],
                [(3, 3), 0, 0.1],
                [(4, 6), 0, 0.15],
                [(7, 10), 1, 0.15],
                [(11, 20), 1, 0.15],
                [(21, 40), 2, 0.15],
                [(41, 80), 2, 0.2],
                [(81, 160), 4, 0.2],
                [(161, 320), 4, 0.25],
                [(321, 1000), 8, 0.25],
            ])

    def split_and_fit(self, random_seed):

        print(random_seed)

        data = self.data

        # Split policy
        new_urm_train = split_with_triple(data.urm_train, self.split_policy)[0]

        # concatenation with ICM
        new_urm_train = sps.vstack([new_urm_train, data.icm_all_augmented.T])

        rec = SLIMElasticNetRecommender(new_urm_train)

        rec.fit(
            topK=self.topK,
            l1_ratio=self.l1_ratio,
            alpha=self.alpha,
            positive_only=self.positive_only,
            max_iter=self.max_iter
        )

        rec.URM_train = data.urm_train

        return rec


class BaggingSLIMRecommender(BaseRecommender):
    RECOMMENDER_NAME = "BaggingSLIMRecommender"

    def __init__(self, data: DataObject):
        super(BaggingSLIMRecommender, self).__init__(data.urm_train)
        self.data = data

    def fit(self,
            number_of_recommender=2,
            split_policy=None,
            topK=100,
            l1_ratio=0.04705,
            alpha=0.00115,
            positive_only=True,
            max_iter=35,
            parallelism=4):

        # topK=100_l1_ratio=0.04705_alpha=0.00115_positive_only=True_max_iter=35.

        # Fit n recommenders
        partial_parallel = par(self.data, split_policy, topK, l1_ratio, alpha, positive_only, max_iter)

        self.recs = Parallel(n_jobs=parallelism)(
            delayed(partial_parallel.split_and_fit)
            (i)
            for i in range(number_of_recommender))




    def _compute_item_score(self, user_id_array, items_to_compute = None):
        # ATTENTION!
        # THIS METHOD WORKS ONLY IF user_id_array IS A SCALAR AND NOT AN ARRAY
        # TODO

        scores = np.zeros(shape=(1, self.n_items))

        for rec in self.recs:
            scores += rec._compute_item_score(user_id_array=user_id_array)

        return scores

