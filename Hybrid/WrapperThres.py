import time

from Base.BaseRecommender import BaseRecommender

from Base.BaseRecommender import BaseRecommender
from DataObject import DataObject
import numpy as np
import random
import pandas as pd
import scipy.sparse as sps
from DataReader import DataReader
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from Hybrid.Hybrid100AlphaRecommender import Hybrid100AlphaRecommender
from Hybrid.Hybrid101AlphaRecommender import Hybrid101AlphaRecommender
from Hybrid.Hybrid102AlphaRecommender import Hybrid102AlphaRecommender
from Hybrid.Hybrid105AlphaRecommender import Hybrid105AlphaRecommender
from Hybrid.Hybrid108AlphaRecommender import Hybrid108AlphaRecommender
from Hybrid.Hybrid109AlphaRecommender import Hybrid109AlphaRecommender
from Hybrid.Hybrid1XXAlphaRecommender import Hybrid1XXAlphaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


class WrapperThres(BaseRecommender):

    ucm_interaction_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_UCM_interaction.csv"
    interaction_df = pd.read_csv(ucm_interaction_path)

    def _get_interactions(self, user_id):
        return self.interaction_df[self.interaction_df['user_id'] == user_id].n_interactions.values[0]

    def __init__(self, data: DataObject, threshold=20, random_seed=1, debug_print=True):
        super(BaseRecommender, self).__init__()
        self.data = data

        print("previously the non zeros were:" + str(data.urm_train.getnnz()))
        self.urm_train_cut = data.urm_train.tolil()
        user_to_cut = data.get_ids_of_train_users_with_more_than_X_interactions(threshold)
        # Auxiliary data structures needed in order to build the csr train matrix
        train_data = []
        train_user = []
        train_item = []

        # Auxiliary data structures needed in order to build the csr train matrix
        hidden_data = []
        hidden_user = []
        hidden_item = []

        # Monitoring time needed for executing the split
        starting_time = time.time()
        if debug_print:
            print("Starting to hid some interactions...")

        # Initializing the random seed
        np.random.seed(random_seed)
        random.seed(random_seed)

        for user_id in user_to_cut:
            n_elem_to_hid = self._get_interactions(user_id) - threshold
            k_elements = random.sample(list(self.data.urm[user_id].indices), n_elem_to_hid) # TODO CHECK
            for item_id in k_elements:
                hidden_data.append(1)
                hidden_user.append(user_id)
                hidden_item.append(item_id)
                self.urm_train_cut[user_id, item_id] = 0

        # Compute the CSR matrix for the train set
        self.urm_train_cut = self.urm_train_cut.tocsr()

        # Compute the CSR matrix for the test set
        self.urm_train_hidden = sps.csr_matrix((hidden_data, (hidden_user, hidden_item)), shape=data.urm.shape)

        print("now the non zeros are:" + str(self.urm_train_cut.getnnz()))
        print("non zeros hidden:" + str(self.urm_train_hidden.getnnz()))
        print("the original urm has:" + str(self.data.urm_train.getnnz()))

        self.rec = RP3betaRecommender(self.urm_train_cut)
        self.rec.fit(topK=30, alpha=0.24, beta=0.24)

    def _remove_seen_on_scores(self, user_id, scores):

        assert self.data.urm_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"

        seen = self.data.urm_train.indices[self.data.urm_train.indptr[user_id]:self.data.urm_train.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores

    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag=False, return_scores=False):


        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.data.urm_train.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self.rec._compute_item_score(user_id_array, items_to_compute=items_to_compute)


        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            if remove_seen_flag:
                scores_batch[user_index,:] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
            # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
            # ranking = relevant_items_partition[relevant_items_partition_sorting]
            #
            # ranking_list.append(ranking)


        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:,0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = [None] * ranking.shape[0]

        # Remove from the recommendation list any item that has a -inf score
        # Since -inf is a flag to indicate an item to remove
        for user_index in range(len(user_id_array)):
            user_recommendation_list = ranking[user_index]
            user_item_scores = scores_batch[user_index, user_recommendation_list]

            not_inf_scores_mask = np.logical_not(np.isinf(user_item_scores))

            user_recommendation_list = user_recommendation_list[not_inf_scores_mask]
            ranking_list[user_index] = user_recommendation_list.tolist()



        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]


        if return_scores:
            return ranking_list, scores_batch

        else:
            return ranking_list


