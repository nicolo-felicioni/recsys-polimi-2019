import random
import numpy as np
import scipy.sparse as sps
from Base.BaseRecommender import BaseRecommender
from DataObject import DataObject
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


class AUserToMultipleUserRecommender(BaseRecommender):
    """AUserToMultipleUserRecommender recommender"""

    RECOMMENDER_NAME = "AUserToMultipleUserRecommender"

    def __init__(self, data: DataObject):
        super(AUserToMultipleUserRecommender, self).__init__(data.urm_train)
        self.data = data
        self.new_urm_train = data.urm_train
        self._generate_new_urm()

    def _generate_new_urm(self, threshold=10):
        data = self.data
        self.users_splitted = data.get_ids_of_train_users_with_more_than_X_interactions(threshold)
        urm = self.new_urm_train
        users_to_split = self.users_splitted
        offset = data.number_of_users
        self.offset = offset
        old_user = []
        old_item = []
        old_data = []
        new_user = []
        new_item = []
        new_data = []
        for user in data.ids_user:
            items = urm[user].indices
            if items.shape[0] > 0:
                items = items.tolist()
                if user in users_to_split:
                    # Get the user interactions
                    n_items_to_be_removed = int(len(items) / 2)
                    removed_items = random.sample(items, n_items_to_be_removed)
                    removed_items_np = np.array(removed_items)
                    removed_items = np.unique(removed_items_np).tolist()
                    for item in items:
                        if item in removed_items:
                            new_user += [user]
                            new_item += [item]
                            new_data += [1]
                        else:
                            old_user += [user]
                            old_item += [item]
                            old_data += [1]
                else:
                    for item in items:
                        old_user += [user]
                        old_item += [item]
                        old_data += [1]
        old_urm_train = sps.csr_matrix((old_data, (old_user, old_item)), shape=urm.shape)
        new_urm_train = sps.csr_matrix((new_data, (new_user, new_item)), shape=urm.shape)
        self.new_urm_train = sps.vstack((old_urm_train, new_urm_train)).tocsr()

    def fit(self, recommender):
        self.recommender = recommender

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        if user_id_array in self.users_splitted:
            r_items_1 = self.recommender.recommend(user_id_array, remove_seen_flag=False)
            r_items_2 = self.recommender.recommend(user_id_array + self.offset, remove_seen_flag=False)
            n = 0
            i = 0
            returned_items = []
            while len(returned_items) < cutoff:
                item = r_items_1[i]
                if item not in self.data.urm_train[user_id_array].indices:
                    returned_items.append(item)
                item = r_items_2[i]
                if item not in self.data.urm_train[user_id_array].indices:
                    returned_items.append(item)
                i = i + 1
            return returned_items[:cutoff]

        else:
            return self.recommender.recommend(user_id_array, cutoff=cutoff)
