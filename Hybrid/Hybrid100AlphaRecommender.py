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
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_AsySVD_Cython
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import numpy as np
import operator


class Hybrid100AlphaRecommender(BaseRecommender):
    """Hybrid100AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid100AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid100AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec1 = UserKNNCBFRecommender(data.ucm_all, data.urm_train)
        self.rec2 = MatrixFactorization_AsySVD_Cython(data.urm_train)
        self.rec1.load_model("Hybrid", "FULL_URM_UserCBF_topK=5000_shrink=5_feature_weighting=TF-IDF_similarity=euclidean")
        self.rec2.load_model("Hybrid", "AsySVD")
        cold = data.ids_cold_user
        train_cold = data.urm_train_users_by_type[0][1]
        # if train_cold.shape[0] > 0:
        #     target_users = np.append(cold, train_cold)
        # else:
        #     target_users = cold
        target_users = data.ids_user
        self.hybrid_rec = Hybrid1CXAlphaRecommender(data, recommenders=[self.rec1, self.rec2],
                                                    recommended_users=target_users, max_cutoff=30)

    def fit(self):
        weights = [
            [0.7609041954308008, 0.27339862890657357, 0.3144084232425596, 0.28678002559795207, 0.3170292366001356,
             0.24147239220026245, 0.14009508172349497, 0.15649320080818696, 0.16743130156581199, 0.18707405544524536,
             0.19350324993686288, 0.21987953364962454, 0.2233212596556791, 0.19217500924897687, 0.19736284590367756,
             0.2100228313521805, 0.18526521644738797, 0.19299238737073587, 0.12892725272565625, 0.12007519721696268,
             0.08916255488946315, 0.09565831116101862, 0.08921835911617365, 0.09575012944298913, 0.06061814295035001,
             0.06788462944119182, 0.037165855793027704, 0.04105324668609332, 0.023576935696646744, 0.0],
            [1.0, 0.23662052508936088, 0.2225837530904237, 0.2536834803952595, 0.2917360024545484, 0.20844068794221393,
             0.11251972620359912, 0.10895216412067295, 0.11738478513380732, 0.10458147363449094, 0.10786698616858342,
             0.11776718227577632, 0.10415827006216491, 0.10285083017703549, 0.11514393572529961, 0.07647362838535736,
             0.08649824797408665, 0.0906695206756817, 0.07494150129954372, 0.06680458369333378, 0.06642942053751508,
             0.06459816503682635, 0.07336768255019717, 0.030660821531648116, 0.027881710157031087, 0.019316732889272565,
             0.02139829161607211, 0.024608035358482924, 0.016094537787800765, 0.010481732651884517]]
        # weights = [[20.75, 12., 15.25, 12.25, 9.75, 12.25, 8.5, 5., 8.5,
        #             9.75, 8.25, 5., 4.75, 7.25, 6.75, 6.25, 8., 8.,
        #             9.5, 7.25, 8.75, 8.25, 6.75, 10.25, 12.25, 8., 8.5,
        #             10., 7.75, 6.5],
        #            [11., 13., 10., 5.25, 7.5, 10., 9.25, 9., 4.,
        #             12.5, 10.75, 11.75, 14., 8.5, 6.5, 6.5, 3.75, 1.75,
        #             6.25, 7.25, 6.25, 8.5, 1.75, 7., 5.5, 6.25, 5.5,
        #             4.25, 3., 4.5]]
        self.hybrid_rec.weights = weights

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        return self.hybrid_rec.recommend(user_id_array=user_id_array, cutoff=cutoff)
