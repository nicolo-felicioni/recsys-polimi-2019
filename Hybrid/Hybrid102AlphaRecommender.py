from Base.BaseRecommender import BaseRecommender
from Base.NonPersonalizedRecommender import TopPop
from DataObject import DataObject
from Data_manager.DataReader import DataReader
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from Hybrid.Hybrid1XXAlphaRecommender import Hybrid1XXAlphaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import numpy as np
import operator


# class Hybrid102AlphaRecommender(BaseRecommender):
#     """Hybrid102AlphaRecommender recommender"""
#
#     RECOMMENDER_NAME = "Hybrid102AlphaRecommender"
#
#     def __init__(self, data: DataObject):
#         super(Hybrid102AlphaRecommender, self).__init__(data.urm_train)
#         self.data = data
#         self.rec1 = SLIM_BPR_Cython(data.urm_train)
#         self.rec2 = ItemKNNCFRecommender(data.urm_train)
#         self.rec3 = RP3betaRecommender(data.urm_train)
#         self.rec1.fit(sgd_mode="adagrad", topK=15000, epochs=250, learning_rate=1e-05, lambda_i=0.01, lambda_j=0.01)
#         self.rec2.fit(topK=10000, shrink=10000, feature_weighting="TF-IDF")
#         self.rec3.fit(topK=10000, alpha=0.55, beta=0.01, implicit=True, normalize_similarity=True)
#         self.hybrid_rec = Hybrid1XXAlphaRecommender(data, recommenders=[self.rec1, self.rec2, self.rec3], max_cutoff=20)
#
#     def fit(self):
#         weights = [[129.1, 66.6, 37.6, 32.3, 21., 15.9, 13.8, 13.3, 10.6, 11.5, 10.7, 10.5,
#                     11.5, 9.7, 8.9, 8.2, 8.5, 8.7, 7.3, 7.4],
#                    [133.9, 66., 41.2, 28., 22.3, 16.7, 15.2, 14., 12.3, 12.2, 12.2, 13.,
#                     13.7, 12.3, 7.2, 9.1, 9.7, 11.6, 8.6, 9.3],
#                    [120.5, 58.3, 36.2, 31.8, 21.1, 18.4, 14.9, 13.2, 15.1, 14.3, 9.1, 9.,
#                     10., 7.9, 9.1, 8.3, 9.7, 6.5, 6.8, 7.4]]
#         self.hybrid_rec.fit(weights=weights)
#
#     def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
#                   remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
#         return self.hybrid_rec.recommend(user_id_array=user_id_array, cutoff=cutoff)

class Hybrid102AlphaRecommender(BaseRecommender):
    """Hybrid101AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid100AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid102AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec1 = ItemKNNCFRecommender(data.urm_train)
        self.rec2 = P3alphaRecommender(data.urm_train)
        # self.rec2 = SLIM_BPR_Cython(data.urm_train)

    def fit(self, coeff=1.4, weights=None):
        # self.rec1.fit(shrink=20000, topK=20000, feature_weighting="TF-IDF")
        self.rec1.fit(shrink=5000, topK=5000, feature_weighting="TF-IDF")
        self.rec2.fit(topK=170, implicit=True, alpha=0.5)
        # self.rec2.fit(epochs=1100, lambda_i=0.001, lambda_j=0.001)
        # self.weights = [
        #     [0.06256913832097202, 0.053942836778445455, 0.031174005992748065, 0.03458361684647441, 0.024475584988981263,
        #      0.03447254341234124, 0.035160531571591505, 0.022996945419287144, 0.020903868144120638,
        #      0.014208195436564087, 0.018615668598092647, 0.02051264219774443, 0.012595658951290738, 0.01632331225411266,
        #      0.009055577269904816, 0.00907152150757566, 0.0069224208081265755, 0.004409374805473207,
        #      0.003119903220847125],
        #     [0.057081432503004376, 0.04033604501530888, 0.02836149952480316, 0.02653191220060894, 0.03586006493158984,
        #      0.036538833372839044, 0.03232448666360269, 0.022689412236584394, 0.017504629606036947,
        #      0.023178613496325582, 0.018595086873988066, 0.013666648028127195, 0.011772826382505389, 0.01703810160286237,
        #      0.011642038211760698, 0.006308243270949454, 0.0073081705576563604, 0.0030333155965175107,
        #      0.0017723789744741641]]
        self.weights = [
            [0.010440085464302622, 0.0070694612566372045, 0.0041928529276911335, 0.0036414239437037956,
             0.0026959008882713794, 0.0043532361543139435, 0.002944005418578649, 0.0031755272886123216,
             0.002634663418865159, 0.0025952979315566056, 0.0005019746815920453, 0.002687225331151587,
             0.0013695165376338798, 0.0014128474025804688, 0.02536392481760321, 0.001037064872535542,
             0.0006348312975730643, 0.0002936218078862895, 0.0003286922859613968],
            [0.011671625532955966, 0.006168717027656684, 0.006066449557313522, 0.004314699731435717,
             0.00495003988915962, 0.0031162066475054332, 0.0036388479392468255, 0.0030901050333030427,
             0.00221359325905791, 0.0028119206953097675, 0.002588751719452233, 0.0026545999438173052,
             0.0014560697487556431, 0.002172643785393432, 0.00021025230689340825, 0.0020337490118575285,
             0.0005797787386621163, 0.00036363289258962983, 0.0003765490781759675]]
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
