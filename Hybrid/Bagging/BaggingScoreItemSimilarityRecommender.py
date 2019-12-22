import copy

from joblib import Parallel, delayed

from Base.BaseRecommender import BaseRecommender
from Base.NonPersonalizedRecommender import TopPop
from DataObject import DataObject, augment_with_item_similarity_best_scores, augment_with_user_similarity_best_scores, \
    augment_with_best_recommended_items
from Data_manager.AdvancedSplitter import split, split_with_triple
from Data_manager.DataReader import DataReader
from Data_manager.Splitter import Splitter
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from Hybrid.Hybrid003AlphaRecommender import Hybrid003AlphaRecommender
from Hybrid.Hybrid100AlphaRecommender import Hybrid100AlphaRecommender
from Hybrid.Hybrid1CXAlphaRecommender import Hybrid1CXAlphaRecommender
from Hybrid.Hybrid1XXAlphaRecommender import Hybrid1XXAlphaRecommender
from Hybrid.Score.HybridItemSimilarityRecommender import HybridItemSimilarityRecommender
from KNN.ItemKNNCBFOnlyColdRecommender import ItemKNNCBFOnlyColdRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import numpy as np
import operator

from SLIM_ElasticNet.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet


def fib(n):
    if n == 0:
        return [0]
    elif n == 1:
        return [0, 1]
    else:
        lst = fib(n - 1)
        lst.append(lst[-1] + lst[-2])
        return lst


class par:
    def __init__(self, data: DataObject, leave_k_out=0, threshold=0, probability=0.2):
        self.data = data
        self.leave_k_out = leave_k_out
        self.threshold = threshold
        self.probability = probability

    def split_and_fit(self, random_seed):
        data = self.data

        # Split policy
        new_urm_train = split_with_triple(data.augmented_urm,
                                          [
                                              [(0, 0), 0, 0.000],
                                              [(1, 2), 0, 0.015],
                                              [(3, 5), 0, 0.030],
                                              [(6, 8), 0, 0.060],
                                              [(9, 11), 0, 0.120],
                                              [(12, 20), 1, 0.240],
                                              [(21, 40), 2, 0.300],
                                              [(41, 80), 3, 0.360],
                                              [(81, 160), 4, 0.420],
                                              [(161, 320), 5, 0.480],
                                              [(321, 1000), 6, 0.540],
                                          ])[0]
        # [
        #     [(0, 0), 0, 0.000],
        #     [(1, 2), 0, 0.015],
        #     [(3, 5), 0, 0.030],
        #     [(6, 8), 0, 0.060],
        #     [(9, 11), 0, 0.120],
        #     [(12, 20), 1, 0.240],
        #     [(21, 40), 2, 0.300],
        #     [(41, 80), 3, 0.360],
        #     [(81, 160), 4, 0.420],
        #     [(161, 320), 5, 0.480],
        #     [(321, 1000), 6, 0.540],
        # ])[0]

        rec = RP3betaRecommender(new_urm_train)
        rec.fit(topK=20, alpha=0.16, beta=0.24)
        rec.URM_train = data.urm_train
        return rec


class BaggingScoreItemSimilarityRecommender(BaseRecommender):
    """BaggingScoreItemSimilarityRecommender recommender"""

    RECOMMENDER_NAME = "BaggingScoreItemSimilarityRecommender"

    def __init__(self, data: DataObject, k: int, leave_k_out=0, threshold=0, probability=0.2):
        super(BaggingScoreItemSimilarityRecommender, self).__init__(data.urm_train)
        self.data = data

        rec = Hybrid100AlphaRecommender(data)
        rec.fit()

        data.augmented_urm = augment_with_best_recommended_items(data.augmented_urm, rec,
                                                                 data.urm_train_users_by_type[0][1], 1, value=0.5)
        data.augmented_urm = augment_with_best_recommended_items(data.augmented_urm, rec,
                                                                 data.urm_train_users_by_type[1][1], 1, value=0.2)
        print(f"After User CBF {data.augmented_urm.nnz}")

        rec = None

        recs = Parallel(n_jobs=6)(
            delayed(par(data,
                        leave_k_out=leave_k_out,
                        threshold=threshold,
                        probability=probability).split_and_fit)
            (i)
            for i in range(k))
        similarities = [rec.W_sparse for rec in recs]
        self.hybrid_rec = HybridItemSimilarityRecommender(data.urm_train)
        self.hybrid_rec.fit(similarities, topK=500)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        return self.hybrid_rec.recommend(user_id_array, cutoff=cutoff)
