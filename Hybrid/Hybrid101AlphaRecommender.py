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
    """Hybrid100AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid100AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid101AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec1 = ItemKNNCFRecommender(data.urm_train)
        self.rec2 = P3alphaRecommender(data.urm_train)
        # self.rec2 = SLIM_BPR_Cython(data.urm_train)

    def fit(self, coeff=1.4, weights=None):
        self.rec1.fit(shrink=20000, topK=20000, feature_weighting="TF-IDF")
        # self.rec1.fit(shrink=16, topK=12, feature_weighting="none")
        self.rec2.fit(topK=170, implicit=True, alpha=0.5)
        # self.rec2.fit(epochs=1100, lambda_i=0.001, lambda_j=0.001)
        self.weights = [
            [0.061619034678667514, 0.06775024725815929, 0.053699659051884605, 0.03720180790336591, 0.039551325666083624,
             0.03229632655215851, 0.022, 0.022, 0.02521690897367572, 0.022452941876168763, 0.027935905298657384,
             0.02785097532053766, 0.015630498840303177, 0.015463976243852737, 0.017481662133569133,
             0.012880841321874269, 0.011284630389699313, 0.006741866068657539, 0.0024840064294532013],
            [0.06647307983109539, 0.034317742164696054, 0.04219442247714708, 0.03293870743176039, 0.03219770719235716,
             0.04636430140428521, 0.040938238487416914, 0.0449188456712402, 0.02563462289211509, 0.02537329610759187,
             0.02493562060910921, 0.01693448418099402, 0.02248408965520954, 0.015491819432766663, 0.01854358542113545,
             0.012310823360234426, 0.013232194443116064, 0.005001098120000997, 0.0021727071109598544]]
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
