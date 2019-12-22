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


class Hybrid104AlphaRecommender(BaseRecommender):
    """Hybrid104AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid104AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid104AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec1 = RP3betaRecommender(data.urm_train)
        self.rec1.fit(topK=26, alpha=0.25, beta=0.21)
        self.rec2 = ItemKNNCFRecommender(data.urm_train)
        self.rec2.fit(topK=10, shrink=1000, similarity="tanimoto", feature_weighting="BM25")
        target_users = data.urm_train_users_by_type[4][1]
        self.target_users = target_users
        self.hybrid_rec = Hybrid1CXAlphaRecommender(data, recommenders=[self.rec1, self.rec2],
                                                    recommended_users=target_users, max_cutoff=30)

    def fit(self):
        weights1 = [
            [1.0, 0.4517179068981683, 0.3979690096061162, 0.4068926692801725, 0.3404408707736778, 0.35438712937353284,
             0.3107497990263178, 0.277027821536975, 0.22460270448263392, 0.258293110155029, 0.22565957216308793,
             0.1427039025093434, 0.14850612508172897, 0.1427281482083753, 0.1513486708877009, 0.14931902151854687,
             0.1717168747463289, 0.0783259533439393, 0.08066358870496233, 0.0799579856421651, 0.06310732697276022,
             0.0679217517962877, 0.07689888001915678, 0.06585136121968413, 0.05989375609327328, 0.05668195370281998,
             0.06377071839974079, 0.017091307253589913, 0.017276230828121276, 0.012865982715760244],
            [0.38688356604557783, 0.3653660032353448, 0.30862315365153425, 0.21699342252527684, 0.24954243590406833,
             0.204684429683601, 0.11665613023016438, 0.12839454425138505, 0.14122535566771405, 0.1504381157229052,
             0.17300383308134096, 0.1309799007170742, 0.1354656505605249, 0.08481274450716023, 0.09302787057145857,
             0.09643918642755862, 0.10731710201865315, 0.07003981095966517, 0.06217937411084906, 0.0457095602617101,
             0.04370382323931054, 0.050259396725207114, 0.05401448061464968, 0.03742468372661131, 0.038730399329483335,
             0.04401751666811602, 0.01793705784582081, 0.020478791947760068, 0.010738842299817633, 0.0]
        ]
        self.hybrid_rec.weights = weights1

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        if user_id_array in self.target_users:
            return self.hybrid_rec.recommend(user_id_array=user_id_array, cutoff=cutoff)
        else:
            return []
