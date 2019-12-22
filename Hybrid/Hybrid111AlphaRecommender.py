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


class Hybrid111AlphaRecommender(BaseRecommender):
    """Hybrid111AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid111AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid111AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec1 = RP3betaRecommender(data.urm_train)
        self.rec2 = ItemKNNCFRecommender(data.urm_train)
        self.rec1.fit(topK=20, alpha=0.11, beta=0.18)
        self.rec2.fit(topK=18, shrink=850, similarity='jaccard', feature_weighting='BM25')
        cold = data.ids_cold_user
        self.target_users = data.urm_train_users_by_type[11][1]
        self.hybrid_rec = Hybrid1CXAlphaRecommender(data, recommenders=[self.rec1, self.rec2],
                                                    recommended_users=self.target_users, max_cutoff=30)

    def fit(self):
        weights = [
            [0.36263544103305234, 0.2821869186577171, 0.2958040731688242, 0.29923612866193294, 0.25220249762628333,
             0.2761756648779422, 0.18915192083976606, 0.182117148228788, 0.1842783740517236, 0.1486885469397763,
             0.15841860217081513, 0.1313525617771801, 0.1495204046677669, 0.1442400061423464, 0.14584743502512043,
             0.12150982415065534, 0.12535649682927805, 0.08906867097805428, 0.1012469101805646, 0.08566556941254282,
             0.08865930444526858, 0.0910510345404526, 0.06557183202757007, 0.05584782961979419, 0.05557650728442622,
             0.030967327271124927, 0.029196638757279396, 0.025490236322759728, 0.01569027879218803,
             0.015055828167915425],
            [1.0, 0.5214104256707622, 0.20746796675006374, 0.23560220633621676, 0.2575770382835032, 0.2512775337476986,
             0.28896916380985344, 0.3287565885723311, 0.19352951261468151, 0.22255893950688374, 0.12074099390243273,
             0.13073528889178715, 0.13251088387516363, 0.13834537392481033, 0.10694898922318279, 0.1211777917594245,
             0.1276279143114306, 0.14661389007809034, 0.14142761528430872, 0.11969443714497587, 0.13398226206599023,
             0.14707641785555184, 0.11689418050769908, 0.0471984606450728, 0.049494500313232624, 0.04680805679550986,
             0.042358887859941775, 0.04342842109287542, 0.0, 0.0]]
        self.hybrid_rec.weights = weights

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        if user_id_array in self.target_users:
            return self.hybrid_rec.recommend(user_id_array=user_id_array, cutoff=cutoff)
        else:
            return []
