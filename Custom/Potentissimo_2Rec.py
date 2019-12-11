from Base.BaseRecommender import BaseRecommender
from DataObject import DataObject
from Hybrid.Hybrid1CXXAlphaRecommender import Hybrid1CXXAlphaRecommender
import numpy as np

class Potentissimo_2Rec(BaseRecommender):
    """Potentissimo_2Rec recommender"""

    RECOMMENDER_NAME = "Potentissimo_2Rec"

    def __init__(self, data : DataObject, rec1 : BaseRecommender, rec2 : BaseRecommender, target_users, max_cutoff=50):
        super(Potentissimo_2Rec, self).__init__(data.urm_train)
        self.target_users = target_users
        self.hybrid_rec = Hybrid1CXXAlphaRecommender(data, recommenders=[rec1, rec2],
                                                    recommended_users=target_users, max_cutoff=max_cutoff)

    def fit(self, w1, w2):
        cutoff_list = [len(w1), len(w2)]
        max_length = max(cutoff_list)
        w1 = np.pad(w1, (0, max_length), 'constant', constant_values=float("-inf"))
        w2 = np.pad(w2, (0, max_length), 'constant', constant_values=float("-inf"))
        weights = np.array((w1, w2))
        self.hybrid_rec.weights = weights
        self.hybrid_rec.cutoff_list = cutoff_list

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        if user_id_array in self.target_users:
            return self.hybrid_rec.recommend(user_id_array=user_id_array, cutoff=cutoff)
        else:
            return []
