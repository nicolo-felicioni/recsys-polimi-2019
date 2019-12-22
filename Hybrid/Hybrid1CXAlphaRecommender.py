import copy

from joblib import Parallel, delayed

from Base.BaseRecommender import BaseRecommender
from DataObject import DataObject
import numpy as np
import operator


def get_cached_recommendation(rec, recommended_users, max_cutoff):
    cached_recommendation = {}
    for user_id in recommended_users:
        recommended_items = rec.recommend(user_id, cutoff=max_cutoff)
        cached_recommendation[user_id] = recommended_items
    return cached_recommendation


class Hybrid1CXAlphaRecommender(BaseRecommender):
    """Hybrid1CXAlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid1CXAlphaRecommender"

    def __init__(self, data: DataObject, recommenders, recommended_users, max_cutoff=20):
        super(Hybrid1CXAlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.weights = []
        self.max_cutoff = max_cutoff
        self.cached_recommendation_all = []
        # for rec in recommenders:
        #     cached_recommendation = {}
        #     for user_id in recommended_users:
        #         recommended_items = rec.recommend(user_id, cutoff=max_cutoff)
        #         cached_recommendation[user_id] = recommended_items
        #     self.cached_recommendation_all.append(cached_recommendation)

        self.cached_recommendation_all = Parallel(n_jobs=8)(
            delayed(get_cached_recommendation)
            (copy.deepcopy(rec), copy.deepcopy(recommended_users), max_cutoff)
            for rec in recommenders)
        for _ in recommenders:
            self.weights.append([x for x in range(2, max_cutoff + 2)][::-1])

    def fit(self):
        pass

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        recommended_items = []
        weighted_item = {}
        limit = self.max_cutoff
        for cached_recommendation in self.cached_recommendation_all:
            recommended_items.append(cached_recommendation[user_id_array])
        for i in range(0, len(recommended_items)):
            for j in range(0, len(recommended_items[i])):
                if j < limit:
                    weighted_item[recommended_items[i][j]] = \
                        weighted_item.get(recommended_items[i][j], 0) - self.weights[i][j]
        result = np.array(sorted(weighted_item.items(), key=operator.itemgetter(1), reverse=False))
        max_size = min(result.shape[0], cutoff)
        if max_size > 0:
            return [int(x) for x in result[:max_size, [0]].squeeze(axis=1).tolist()]
        else:
            return np.array([])

    def clone(self):
        clone_of_this = copy.copy(self)
        return clone_of_this
