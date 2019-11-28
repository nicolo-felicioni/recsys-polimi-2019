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


def fib(n):
    if n == 0:
        return [0]
    elif n == 1:
        return [0, 1]
    else:
        lst = fib(n - 1)
        lst.append(lst[-1] + lst[-2])
        return lst


class Hybrid109AlphaRecommender(BaseRecommender):
    """Hybrid109AlphaRecommender recommender"""

    RECOMMENDER_NAME = "Hybrid109AlphaRecommender"

    def __init__(self, data: DataObject):
        super(Hybrid109AlphaRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec1 = ItemKNNCFRecommender(data.urm_train)
        self.rec2 = P3alphaRecommender(data.urm_train)
        self.rec3 = RP3betaRecommender(data.urm_train)
        self.rec1.fit(shrink=15, topK=12, feature_weighting="none")
        self.rec2.fit(topK=170, implicit=True, alpha=0.5)
        self.rec3.fit(topK=60, alpha=0.5, beta=0.1, implicit=True)
        self.hybrid_rec = Hybrid1XXAlphaRecommender(data, recommenders=[self.rec1, self.rec2, self.rec3], max_cutoff=12)

    def fit(self):
        # weights = [[0.011266893152536605, 0.007848694733480117, 0.00678606758121504, 0.0006067954946807141,
        #             0.006426152115795074, 0.0002300635534871915, 0.0051536160070757365, 0.00034042295154708776,
        #             0.00022004652352997558, 0.011096242658341115, 0.00018224078726942006, 0.0029205736537706,
        #             0.0048435410092260035, 0.002974170299608932, 0.07739093126537824, 0.0018468641905625227,
        #             9.836028669311674e-05, 0.00158419008013134, 0.0007734766378183035],
        #            [0.009738474365252065, 0.0072219319338005985, 0.010321437262331335, 0.006592984860762474,
        #             0.005906701428340946, 0.0054300945219053, 0.0063001654831622, 0.0037592932154427807,
        #             0.003989673349816449, 0.0033627778386283312, 0.004865485128359383, 0.004385062567450715,
        #             0.05224443995791531, 0.0033722844422431576, 3.2694518709465355e-06, 0.001725927418254763,
        #             0.001059313848103872, 0.001040295332587453, 2.3274450522501288e-05],
        #            [0.5705515574517951, 0.025496904603501257, 0.011299185926415923, 0.013297834292548338,
        #             0.007652401893264896, 0.010812688545040892, 0.012751011149275251, 0.009419284302642286,
        #             0.013118400580623218, 0.012701448066742504, 0.006214618815426535, 0.00023940247582970978,
        #             0.009958235830202045, 0.007368788409660027, 0.004424112950571024, 0.004486417711812658,
        #             0.004395442776040509, 0.0010352345795761636, 0.0008447998001340454]]
        weights = [[0.043711522288854034, 0.0051960719619208165, 0.05667484501683396, 0.030629545038921412,
                    0.012197521070395888, 0.019477804210941762, 0.00738781569312461, 0.011723825327847274,
                    0.0028121988564594577, 0.0005813958343011877, 0.007503820009874737, 0.03309879187707586,
                    0.0002054099498172143, 0.00014997422616167206, 0.0004090074650382679, 0.0018478068660466686,
                    0.0016577479487689126, 0.0012228172396937084, 0.0025165843431409055],
                   [3.9057130473851296e-05, 0.011159558038984338, 0.0052683623724797, 0.004826699812159992,
                    0.00042334184507125677, 0.016162397034570956, 0.00016330899098966576, 0.0009324832211376287,
                    0.0013009689619712858, 0.0014913072873020607, 0.019907764218705708, 0.007141115308324247,
                    0.00252244692443465, 0.0065864320607641165, 0.0038287952668927838, 0.0027865290411239123,
                    0.0022007666628621916, 0.00107638276749647, 2.1728950576793545e-06],
                   [0.3131117420253725, 0.061240955842440305, 0.039237684292952796, 0.011052660007541033,
                    0.02205865459222238, 0.03339117117573062, 0.02896461995545859, 0.04729472823177208,
                    0.012766832446648349, 0.02125587453585608, 0.021421997219138792, 0.011940423618529322,
                    0.0008352373940102355, 0.003992857231550777, 0.0015931555090141158, 0.013079464730993284,
                    0.02629003636815483, 0.0032641864791993955, 0.00038332327739396026]]
        self.hybrid_rec.fit(weights=weights)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        return self.hybrid_rec.recommend(user_id_array=user_id_array, cutoff=cutoff)
