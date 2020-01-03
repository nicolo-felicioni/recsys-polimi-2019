import copy

from joblib import Parallel, delayed

from Base.BaseRecommender import BaseRecommender
from Base.NonPersonalizedRecommender import TopPop
from DataObject import DataObject, augment_with_item_similarity_best_scores, augment_with_user_similarity_best_scores, \
    augment_with_best_recommended_items
from Data_manager.AdvancedSplitter import split, split_with_triple
from Hybrid.Score.HybridItemSimilarityRecommender import HybridItemSimilarityRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import numpy as np

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
    def __init__(self,
                 data: DataObject,
                 split_policy=None,
                 topK=22,
                 shrink=850,
                 similarity="tanimoto",
                 feature_weighting="TF_IDF"
                 ):
        self.data = data
        self.split_policy = split_policy
        self.topK = topK
        self.shrink = shrink
        self.similarity = similarity
        self.feature_weighting = feature_weighting

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
        new_urm_train = split_with_triple(data.augmented_urm, self.split_policy)[0]

        rec = ItemKNNCFRecommender(new_urm_train)

        rec.fit(
            topK=self.topK,
            shrink=self.shrink,
            similarity=self.similarity,
            feature_weighting=self.feature_weighting
        )

        rec.URM_train = data.urm_train

        return rec


class BaggingItemCFRecommender(BaseRecommender):
    RECOMMENDER_NAME = "BaggingItemCFRecommender"

    def __init__(self, data: DataObject):
        super(BaggingItemCFRecommender, self).__init__(data.urm_train)
        self.data = data
        self.hybrid_rec = HybridItemSimilarityRecommender(self.data.urm_train)

    def fit(self,
            number_of_recommender=2,
            split_policy=None,
            topK=22,
            shrink=850,
            similarity="tanimoto",
            feature_weighting="TF_IDF",
            hybrid_topK=20,
            parallelism=4):

        # Fit n recommenders
        partial_parallel = par(self.data, split_policy, topK, shrink, similarity, feature_weighting)
        recs = Parallel(n_jobs=parallelism)(
            delayed(partial_parallel.split_and_fit)
            (i)
            for i in range(number_of_recommender))
        similarities = [rec.W_sparse for rec in recs]

        # Sum the similarity matrices
        self.hybrid_rec = HybridItemSimilarityRecommender(self.data.urm_train)
        self.hybrid_rec.fit(similarities, topK=hybrid_topK)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        return self.hybrid_rec.recommend(user_id_array, cutoff=cutoff)

    def save(self, random_seed):
        self.hybrid_rec.save_model("BaggingItemCFRecommender", f"{random_seed}")

    def load(self, random_seed):
        self.hybrid_rec.load_model("BaggingItemCFRecommender", f"{random_seed}")