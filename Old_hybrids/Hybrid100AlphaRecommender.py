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
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import numpy as np
import operator


# class Hybrid100AlphaRecommender(BaseRecommender):
#     """Hybrid100AlphaRecommender recommender"""
#
#     RECOMMENDER_NAME = "Hybrid100AlphaRecommender"
#
#     def __init__(self, data: DataObject):
#         super(Hybrid100AlphaRecommender, self).__init__(data.urm_train)
#         self.data = data
#         self.rec1 = UserKNNCBFRecommender(data.ucm_all, data.urm_train)
#         self.rec2 = TopPop(data.urm_train)
#         self.rec1.fit(shrink=1, topK=11000)
#         self.rec2.fit()
#         cold = data.ids_cold_user
#         train_cold = data.urm_train_users_by_type[0][1]
#         if train_cold.shape[0] > 0:
#             target_users = np.append(cold, train_cold)
#         else:
#             target_users = cold
#         self.hybrid_rec = Hybrid1CXAlphaRecommender(data, recommenders=[self.rec1, self.rec2],  recommended_users=target_users, max_cutoff=20)
#
#     def fit(self):
#         weights = [[19.53, 11.84, 14.84, 12.36, 12.56, 12.07, 7.57, 6.79, 7.47, 7.12, 7.74, 5.74,
#                     5.62, 5.99, 7.04, 7.21, 7.58, 8.72, 9.63, 9.29, 8.82, 8.5, 8.29, 8.28,
#                     9.65, 9.17, 9.76, 8.32, 7.03, 6.96],
#                    [11.02, 12.8, 10.18, 7.3, 7.65, 8.82, 8.68, 8.39, 4.88, 11.28, 11.68, 11.02,
#                     9.61, 9.07, 7.89, 6.83, 4.84, 3.11, 4.78, 4.37, 5.63, 6.84, 8.22, 6.08,
#                     5.8, 6.83, 6.37, 3.64, 4.04, 6.]]
#         # weights = [[20.75, 12., 15.25, 12.25, 9.75, 12.25, 8.5, 5., 8.5,
#         #             9.75, 8.25, 5., 4.75, 7.25, 6.75, 6.25, 8., 8.,
#         #             9.5, 7.25, 8.75, 8.25, 6.75, 10.25, 12.25, 8., 8.5,
#         #             10., 7.75, 6.5],
#         #            [11., 13., 10., 5.25, 7.5, 10., 9.25, 9., 4.,
#         #             12.5, 10.75, 11.75, 14., 8.5, 6.5, 6.5, 3.75, 1.75,
#         #             6.25, 7.25, 6.25, 8.5, 1.75, 7., 5.5, 6.25, 5.5,
#         #             4.25, 3., 4.5]]
#         self.hybrid_rec.weights = weights
#
#     def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
#                   remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
#         return self.hybrid_rec.recommend(user_id_array=user_id_array, cutoff=cutoff)


class Hybrid100AlphaRecommender(BaseRecommender):
    """Hybrid100AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid100AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid100AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec1 = UserKNNCBFRecommender(data.ucm_all, data.urm_train)
        self.rec2 = TopPop(data.urm_train)
        self.rec1.fit(topK=5000, shrink=5, feature_weighting="TF-IDF", similarity="euclidean")
        self.rec2.fit()
        cold = data.ids_cold_user
        train_cold = data.urm_train_users_by_type[0][1]
        if train_cold.shape[0] > 0:
            target_users = np.append(cold, train_cold)
        else:
            target_users = cold
        self.target_users = target_users
        self.hybrid_rec = Hybrid1CXAlphaRecommender(data, recommenders=[self.rec1, self.rec2],
                                                    recommended_users=target_users, max_cutoff=20)

    def fit(self):
        weights = [
            [1.0, 0.6821568252468901, 0.6849024969656253, 0.5278884115166208, 0.5256279588345396, 0.3523944402333277,
             0.35917072524518867, 0.3781483902477164, 0.35494012640675526, 0.2569761057765945, 0.2686708563615972,
             0.26047695765786827, 0.18963922521929752, 0.19796884202620682, 0.21062192323921447, 0.21086230330374983,
             0.20645211181144751, 0.2051545899987307, 0.195630375727827, 0.16433006044601312, 0.15898416104687418,
             0.13710408424057147, 0.14535409877804073, 0.14093706131152708, 0.13925057012140438, 0.10897116615151631,
             0.08690310920352357, 0.05452137761633189, 0.039626490480693964, 0.038884188246590415],
            [0.12184895015208339, 0.10412792995227525, 0.11245816434845729, 0.09237425432552643, 0.07790937226904884,
             0.08414212205057275, 0.06899821248751226, 0.07451806948651324, 0.06080223143736178, 0.05923095707738565,
             0.0639694336435765, 0.06890447942434409, 0.06474046677091937, 0.0672866612162612, 0.06627995889505314,
             0.06170959010489348, 0.05576908741213118, 0.05516680106058056, 0.04913346628366165, 0.042346032068023606,
             0.04268540340456829, 0.03738588109631485, 0.03801703741074155, 0.03634930102905383, 0.027148968808256298,
             0.023744038897221154, 0.019516056544059187, 0.009815556133702266, 0.0038914891192522562,
             0.0034155098002848494]]
        self.hybrid_rec.weights = weights

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        if user_id_array in self.target_users:
            return self.hybrid_rec.recommend(user_id_array=user_id_array, cutoff=cutoff)
        else:
            return []
