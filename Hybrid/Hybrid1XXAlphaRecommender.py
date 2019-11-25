from Base.BaseRecommender import BaseRecommender
from DataObject import DataObject
from Data_manager.DataReader import DataReader
from GraphBased.P3alphaRecommender import P3alphaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import numpy as np
import operator
def fib(n):
    if n == 0:
        return [0]
    elif n == 1:
        return [0, 1]
    else:
        lst = fib(n-1)
        lst.append(lst[-1] + lst[-2])
        return lst

class Hybrid1XXAlphaRecommender(BaseRecommender):
    """Hybrid1XXAlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid1XXAlphaRecommender"

    def __init__(self, data: DataObject, recommenders):
        super(Hybrid1XXAlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.recommenders = recommenders
        self.weights = []
        self.coeff = 1.2

    def fit(self, coeff=1.2, weights=None):
        self.weights = weights
        self.coeff=coeff

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        recommended_items = []
        weighted_item = {}
        for rec in self.recommenders:
            recommended_items.append(np.array(rec.recommend(user_id_array, cutoff=int(cutoff * self.coeff))))
        for i in range(0, len(recommended_items)):
            for j in range(0, len(recommended_items[i])):
                weighted_item[recommended_items[i][j]] =\
                    weighted_item.get(recommended_items[i][j], 0) - self.weights[i][j]
        result = np.array(sorted(weighted_item.items(), key=operator.itemgetter(1), reverse=False))
        max_size = min(result.shape[0], cutoff)
        if max_size > 0:
            return [int(x) for x in result[:max_size, [0]].squeeze(axis=1).tolist()]
        else:
            return np.array([])
