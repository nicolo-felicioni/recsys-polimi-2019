import copy

from Data_manager.Splitter import Splitter
import numpy as np
from scipy import sparse as sps

class DataObject(object):

    def __init__(self, data_reader):
        self.data_reader = data_reader
        self.urm, self.ids_warm_user, self.ids_warm_item = data_reader.load_urm()
        self.number_of_users = self.urm.shape[0]
        self.number_of_items = self.urm.shape[1]
        self.ids_user = range(0, self.number_of_users)
        self.ids_item = range(0, self.number_of_items)
        self.ids_cold_user = [user for user in range(0, self.number_of_users) if user not in self.ids_warm_user]
        self.ids_cold_item = [item for item in range(0, self.number_of_items) if item not in self.ids_warm_item]
        self.number_of_warm_users = self.ids_warm_user.shape[0]
        self.number_of_warm_users = self.ids_warm_item.shape[0]
        self.number_of_cold_users = len(self.ids_cold_user)
        self.number_of_cold_items = len(self.ids_cold_item)
        self.ids_target_users = data_reader.load_target()
        self.number_of_target_users = len(self.ids_target_users)
        self.icm_asset = data_reader.load_icm_asset()
        self.icm_price = data_reader.load_icm_price()
        self.icm_class = data_reader.load_icm_class()
        splitter = Splitter(self.urm)
        splitter.split_train_test(k=1, probability=0, random_seed=17)
        self.urm_train = splitter.train_csr
        self.urm_test = splitter.test_csr
        self.ids_warm_train_users = splitter.ids_warm_train_users
        self.ids_warm_train_items = splitter.ids_warm_train_items
        self.ids_cold_train_users = splitter.ids_cold_train_users
        self.ids_cold_train_items = splitter.ids_cold_train_items
        self.number_of_warm_train_users = splitter.number_of_warm_train_users
        self.number_of_warm_train_items = splitter.number_of_warm_train_items
        self.number_of_cold_train_users = splitter.number_of_cold_train_users
        self.number_of_cold_train_items = splitter.number_of_cold_train_items

    def clone(self):
        return copy.deepcopy(self)

    def print(self):
        print(f"urm: {self.urm}"
              f"urm size: {self.urm.shape}"
              f"number of users: {self.number_of_users}"
              f"number of items: {self.number_of_items}"
              f"")

    def print_statistics(self):
        n_users = self.n_users
        n_items = self.n_items
        n_interactions = len(self.urm.data)
        URM_all = self.urm

        user_profile_length = np.ediff1d(URM_all.indptr)

        max_interactions_per_user = user_profile_length.max()
        avg_interactions_per_user = n_interactions / n_users
        min_interactions_per_user = user_profile_length.min()

        URM_all = sps.csc_matrix(URM_all)
        item_profile_length = np.ediff1d(URM_all.indptr)

        max_interactions_per_item = item_profile_length.max()
        avg_interactions_per_item = n_interactions / n_items
        min_interactions_per_item = item_profile_length.min()

        print("DataReader: current dataset is: {}\n"
              "\tNumber of items: {}\n"
              "\tNumber of users: {}\n"
              "\tNumber of interactions in URM_all: {}\n"
              "\tInteraction density: {:.2E}\n"
              "\tInteractions per user:\n"
              "\t\t Min: {:.2E}\n"
              "\t\t Avg: {:.2E}\n"
              "\t\t Max: {:.2E}\n"
              "\tInteractions per item:\n"
              "\t\t Min: {:.2E}\n"
              "\t\t Avg: {:.2E}\n"
              "\t\t Max: {:.2E}\n".format(
            self.__class__,
            n_items,
            n_users,
            n_interactions,
            n_interactions / (n_items * n_users),
            min_interactions_per_user,
            avg_interactions_per_user,
            max_interactions_per_user,
            min_interactions_per_item,
            avg_interactions_per_item,
            max_interactions_per_item,
            #gini_index(user_profile_length),
        ))