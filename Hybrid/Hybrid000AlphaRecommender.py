from Base.BaseRecommender import BaseRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


class Hybrid000AlphaRecommender(BaseRecommender):
    """Random recommender"""

    RECOMMENDER_NAME = "RandomRecommender"

    def __init__(self, URM_train, UCM, cold_users, warm_users):
        super(Hybrid000AlphaRecommender, self).__init__(URM_train)
        self.warm_recommender = SLIM_BPR_Cython(URM_train)
        self.cold_recommender = UserKNNCBFRecommender(UCM, URM_train)
        self.cold_users = cold_users
        self.warm_users = warm_users


    def fit(self, random_seed=42, epochs=1200, topK = 10000, shrink=1):
        self.warm_recommender.fit(epochs=1100)
        self.cold_recommender.fit(topK=topK, shrink=shrink)


    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_CustomItems_flag = False, return_scores = False):
        if user_id_array in self.warm_users:
            return self.warm_recommender.recommend(user_id_array, cutoff=cutoff)
        if user_id_array in self.cold_users:
            return self.cold_recommender.recommend(user_id_array, cutoff=cutoff)