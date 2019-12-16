from Base.BaseRecommender import BaseRecommender
from DataObject import DataObject
from Hybrid.Hybrid1CXXAlphaRecommender import Hybrid1CXXAlphaRecommender
import numpy as np

class ImplicitRecommender(BaseRecommender):
    """ImplicitRecommender recommender"""

    RECOMMENDER_NAME = "ImplicitRecommender"

    def __init__(self, data : DataObject, rec):
        super(ImplicitRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec = rec
        self.user_items = data.urm_train

    def fit(self):
        pass

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        # items_to_be_recommended = [x
        #                            for x in self.data.ids_item
        #                            if x not in self.data.urm_train[user_id_array].indices]
        return self.rec.recommend(user_id_array, self.user_items, filter_already_liked_items=True)

