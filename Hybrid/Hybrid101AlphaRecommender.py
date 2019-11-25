from Base.BaseRecommender import BaseRecommender
from Base.NonPersonalizedRecommender import TopPop
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
        lst = fib(n - 1)
        lst.append(lst[-1] + lst[-2])
        return lst


class Hybrid101AlphaRecommender(BaseRecommender):
    """Hybrid101AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid100AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid101AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec1 = ItemKNNCFRecommender(data.urm_train)
        self.rec2 = P3alphaRecommender(data.urm_train)
        # self.rec2 = SLIM_BPR_Cython(data.urm_train)

    def fit(self, coeff=1.4, weights=None):
        # self.rec1.fit(shrink=20000, topK=20000, feature_weighting="TF-IDF")
        self.rec1.fit(shrink=15, topK=12, feature_weighting="none")
        self.rec2.fit(topK=170, implicit=True, alpha=0.5)
        # self.rec2.fit(epochs=1100, lambda_i=0.001, lambda_j=0.001)
        self.weights = [
            [0.06256913832097202, 0.053942836778445455, 0.031174005992748065, 0.03458361684647441, 0.024475584988981263,
             0.03447254341234124, 0.035160531571591505, 0.022996945419287144, 0.020903868144120638,
             0.014208195436564087, 0.018615668598092647, 0.02051264219774443, 0.012595658951290738, 0.01632331225411266,
             0.009055577269904816, 0.00907152150757566, 0.0069224208081265755, 0.004409374805473207,
             0.003119903220847125],
            [0.057081432503004376, 0.04033604501530888, 0.02836149952480316, 0.02653191220060894, 0.03586006493158984,
             0.036538833372839044, 0.03232448666360269, 0.022689412236584394, 0.017504629606036947,
             0.023178613496325582, 0.018595086873988066, 0.013666648028127195, 0.011772826382505389, 0.01703810160286237,
             0.011642038211760698, 0.006308243270949454, 0.0073081705576563604, 0.0030333155965175107,
             0.0017723789744741641]]
        self.coeff = coeff

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        items1 = np.array(self.rec1.recommend(user_id_array, cutoff=int(cutoff * self.coeff)))
        items2 = np.array(self.rec2.recommend(user_id_array, cutoff=int(cutoff * self.coeff)))
        weighted_item = {}
        for i in range(0, items1.shape[0]):
            weighted_item[items1[i]] = weighted_item.get(items1[i], 0) - self.weights[0][i]
        for i in range(0, items2.shape[0]):
            weighted_item[items2[i]] = weighted_item.get(items2[i], 0) - self.weights[1][i]
        result = np.array(sorted(weighted_item.items(), key=operator.itemgetter(1), reverse=False))
        max_size = min(result.shape[0], cutoff)
        if (max_size > 0):
            return [int(x) for x in result[:max_size, [0]].squeeze(axis=1).tolist()]
        else:
            return np.array([])