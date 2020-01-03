
from Base.BaseRecommender import BaseRecommender
from Custom.ImplicitBaseRecommender import ImplicitBaseRecommender
from DataObject import DataObject
from Hybrid.Hybrid1CXXAlphaRecommender import Hybrid1CXXAlphaRecommender
import numpy as np
import implicit


class ImplicitNMSLibRecommender(ImplicitBaseRecommender):
    """ImplicitNMSLibRecommender recommender"""

    RECOMMENDER_NAME = "ImplicitNMSLibRecommender"

    def __init__(self, data : DataObject,factors=100, regularization=0.01, learning_rate=1e-3,
                                                        use_gpu=False, iterations=15, num_threads=0):

        super(ImplicitNMSLibRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec = implicit.approximate_als.NMSLibAlternatingLeastSquares(iterations=50)

    def fit(self, show_progress=True):
        self.rec.fit(Ciu=self.data.urm_train.T)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_not_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):

        list_tuples_item_score = self.rec.recommend(userid=user_id_array, user_items=self.URM_train,
                                                    N=cutoff)

        if (return_scores):
            return list_tuples_item_score
        else:
            list_items = []
            for tuple in list_tuples_item_score:
                item = tuple[0]
                list_items.append(item)
            return list_items
