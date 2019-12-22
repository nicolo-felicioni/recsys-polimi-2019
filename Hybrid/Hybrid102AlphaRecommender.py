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
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
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


# LAST OLD (NEWEST)
# class Hybrid102AlphaRecommender(BaseRecommender):
#     """Hybrid101AlphaRecommender recommender"""
#
#     RECOMMENDER_NAME = "Hybrid100AlphaRecommender"
#
#     def __init__(self, data: DataObject):
#         super(Hybrid102AlphaRecommender, self).__init__(data.urm_train)
#         self.data = data
#         self.rec1 = ItemKNNCFRecommender(data.urm_train)
#         self.rec2 = P3alphaRecommender(data.urm_train)
#         # self.rec2 = SLIM_BPR_Cython(data.urm_train)
#
#     def fit(self, coeff=1.4, weights=None):
#         # self.rec1.fit(shrink=20000, topK=20000, feature_weighting="TF-IDF")
#         self.rec1.fit(shrink=5000, topK=5000, feature_weighting="TF-IDF")
#         self.rec2.fit(topK=170, implicit=True, alpha=0.5)
#         # self.rec2.fit(epochs=1100, lambda_i=0.001, lambda_j=0.001)
#         # self.weights = [
#         #     [0.06256913832097202, 0.053942836778445455, 0.031174005992748065, 0.03458361684647441, 0.024475584988981263,
#         #      0.03447254341234124, 0.035160531571591505, 0.022996945419287144, 0.020903868144120638,
#         #      0.014208195436564087, 0.018615668598092647, 0.02051264219774443, 0.012595658951290738, 0.01632331225411266,
#         #      0.009055577269904816, 0.00907152150757566, 0.0069224208081265755, 0.004409374805473207,
#         #      0.003119903220847125],
#         #     [0.057081432503004376, 0.04033604501530888, 0.02836149952480316, 0.02653191220060894, 0.03586006493158984,
#         #      0.036538833372839044, 0.03232448666360269, 0.022689412236584394, 0.017504629606036947,
#         #      0.023178613496325582, 0.018595086873988066, 0.013666648028127195, 0.011772826382505389, 0.01703810160286237,
#         #      0.011642038211760698, 0.006308243270949454, 0.0073081705576563604, 0.0030333155965175107,
#         #      0.0017723789744741641]]
#         self.weights = [
#             [0.010440085464302622, 0.0070694612566372045, 0.0041928529276911335, 0.0036414239437037956,
#              0.0026959008882713794, 0.0043532361543139435, 0.002944005418578649, 0.0031755272886123216,
#              0.002634663418865159, 0.0025952979315566056, 0.0005019746815920453, 0.002687225331151587,
#              0.0013695165376338798, 0.0014128474025804688, 0.02536392481760321, 0.001037064872535542,
#              0.0006348312975730643, 0.0002936218078862895, 0.0003286922859613968],
#             [0.011671625532955966, 0.006168717027656684, 0.006066449557313522, 0.004314699731435717,
#              0.00495003988915962, 0.0031162066475054332, 0.0036388479392468255, 0.0030901050333030427,
#              0.00221359325905791, 0.0028119206953097675, 0.002588751719452233, 0.0026545999438173052,
#              0.0014560697487556431, 0.002172643785393432, 0.00021025230689340825, 0.0020337490118575285,
#              0.0005797787386621163, 0.00036363289258962983, 0.0003765490781759675]]
#         self.coeff = coeff
#
#     def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
#                   remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
#         items1 = np.array(self.rec1.recommend(user_id_array, cutoff=int(cutoff * self.coeff)))
#         items2 = np.array(self.rec2.recommend(user_id_array, cutoff=int(cutoff * self.coeff)))
#         weighted_item = {}
#         for i in range(0, items1.shape[0]):
#             weighted_item[items1[i]] = weighted_item.get(items1[i], 0) - self.weights[0][i]
#         for i in range(0, items2.shape[0]):
#             weighted_item[items2[i]] = weighted_item.get(items2[i], 0) - self.weights[1][i]
#         result = np.array(sorted(weighted_item.items(), key=operator.itemgetter(1), reverse=False))
#         max_size = min(result.shape[0], cutoff)
#         if (max_size > 0):
#             return [int(x) for x in result[:max_size, [0]].squeeze(axis=1).tolist()]
#         else:
#             return np.array([])

class Hybrid102AlphaRecommender(BaseRecommender):
    """Hybrid102AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid102AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid102AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec1 = UserKNNCFRecommender(data.urm_train)
        self.rec1.fit(topK=1000, shrink=4500, similarity="cosine", feature_weighting="TF-IDF")
        self.rec2 = ItemKNNCFRecommender(data.urm_train)
        self.rec2.fit(topK=2000, shrink=800, similarity="cosine", feature_weighting="TF-IDF")
        self.rec3 = SLIM_BPR_Cython(data.urm_train)
        self.rec3.fit(epochs=120, topK=800, lambda_i=0.1, lambda_j=0.1, learning_rate=0.0001)
        self.rec4 = RP3betaRecommender(data.urm_train)
        self.rec4.fit(topK=30, alpha=0.21, beta=0.25)
        target_users = data.urm_train_users_by_type[2][1]
        self.target_users = target_users
        self.hybrid_rec = Hybrid1CXAlphaRecommender(data, recommenders=[self.rec1, self.rec2, self.rec3],
                                                    recommended_users=target_users, max_cutoff=30)

    def fit(self):
        weights1 = np.array([[0.37625119, 0.43193487, 0.17444842, 0.16197883, 0.18204363,
                              0.17016599, 0.14983434, 0.11938279, 0.09980418, 0.1147748,
                              0.12762677, 0.08689066, 0.09533745, 0.10492991, 0.097475,
                              0.05278562, 0.05244627, 0.0602501, 0.06743845, 0.06145589,
                              0.07008017, 0.07410305, 0.07170746, 0.04231058, 0.04493697,
                              0.02516579, 0.0176046, 0.01360429, 0., 0.],
                             [0.55298149, 0.27456885, 0.2278579, 0.25095311, 0.24721051,
                              0.09937549, 0.09209609, 0.07158969, 0.07174988, 0.08251237,
                              0.09157335, 0.10530935, 0.1106961, 0.12150468, 0.12001527,
                              0.10052318, 0.09536568, 0.10770821, 0.08553278, 0.06198749,
                              0.05708056, 0.05176975, 0.05953521, 0.05567152, 0.06083775,
                              0.02776653, 0.02663699, 0.01181728, 0.01168978, 0.],
                             [0.25041731, 0.15536414, 0.16953122, 0.17164006, 0.11443169,
                              0.11873982, 0.07100542, 0.06452205, 0.06123626, 0.06430055,
                              0.06311274, 0.05618836, 0.05331187, 0.04611177, 0.04239514,
                              0.03824963, 0.04398116, 0.04738213, 0.04862799, 0.03962175,
                              0.04556502, 0.04738956, 0.054498, 0.0626727, 0.04973429,
                              0.03219802, 0.03227312, 0.0307041, 0.03396853, 0.],
                             [1., 0.5538608, 0.44692181, 0.20321725, 0.22012478,
                              0.1873366, 0.14329206, 0.09783222, 0.10765581, 0.10658318,
                              0.12257066, 0.13699397, 0.15743225, 0.12181424, 0.13897041,
                              0.08672218, 0.09188654, 0.05170634, 0.04459521, 0.04785834,
                              0.05248675, 0.06035977, 0.06733202, 0.06760871, 0.07775002,
                              0.0720465, 0.05977294, 0.04260028, 0.00546561, 0.0055422]])
        self.hybrid_rec.weights = weights1

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        if user_id_array in self.target_users:
            return self.hybrid_rec.recommend(user_id_array=user_id_array, cutoff=cutoff)
        else:
            return []
