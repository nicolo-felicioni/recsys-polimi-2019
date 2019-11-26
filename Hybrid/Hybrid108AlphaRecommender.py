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


class Hybrid108AlphaRecommender(BaseRecommender):
    """Hybrid108AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid108AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid108AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec1 = ItemKNNCFRecommender(data.urm_train)
        self.rec2 = P3alphaRecommender(data.urm_train)
        self.rec3 = RP3betaRecommender(data.urm_train)
        self.rec1.fit(shrink=15, topK=12, feature_weighting="none")
        self.rec2.fit(topK=170, implicit=True, alpha=0.5)
        self.rec3.fit(topK=60, alpha=0.5, beta=0.1, implicit=True)
        self.hybrid_rec = Hybrid1XXAlphaRecommender(data, recommenders=[self.rec1, self.rec2, self.rec3])

    def fit(self):
        weights = [[0.013955793133815448, 0.016535437471757994, 0.020879904694131795, 0.018819112714618096,
                    0.021392445706548212, 0.01607548981155874, 0.012738476492879265, 0.009777266742758133,
                    0.011917968894846576, 0.016232331241573205, 0.00044031807285253724, 0.010311619128170772,
                    0.0003174819105500035, 0.005841841095482168, 0.13107594580511955, 0.006823499030831328,
                    0.0001859747790325677, 0.00013628669272766473, 0.043318992290746205],
                   [0.03356288888556048, 0.021089291782058508, 0.027709259885623092, 0.011868410835421112,
                    0.00973341821458705, 0.0161198078834219, 0.015418120586074578, 0.0003829496428369569,
                    0.010574899655600982, 0.015410956572233233, 0.012972558536418256, 0.012442773171366942,
                    0.008062770820531412, 0.006539198725414654, 0.0030793859641297097, 0.004638078871153262,
                    0.04670122729818352, 0.04669890489407586, 1.5519813457027137e-06],
                   [0.056308728999257596, 0.041239618977101836, 0.028082688269840384, 0.032933230417959626,
                    0.0016038882236282868, 0.023466868544586666, 0.0015812500911792805, 0.021817296709852694,
                    0.020399082356084723, 0.020807427232841697, 0.013770647079198306, 0.007632514947152459,
                    0.02224168668294638, 0.014297207549573789, 0.012705959058703952, 0.008989890821227529,
                    0.00837759285533516, 0.002475224671966985, 0.0014865565955251984]]
        self.hybrid_rec.fit(weights=weights)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        return self.hybrid_rec.recommend(user_id_array=user_id_array, cutoff=cutoff)
