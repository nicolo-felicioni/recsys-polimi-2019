import copy

from Data_manager.Splitter import Splitter
import numpy as np
from scipy import sparse as sps


class DataObject(object):

    def __init__(self, data_reader, k, random_seed=16):
        self.data_reader = data_reader
        self.urm, self.ids_warm_user, self.ids_warm_item = data_reader.load_urm()
        self.number_of_users = self.urm.shape[0]
        self.number_of_items = self.urm.shape[1]
        self.ids_user = range(0, self.number_of_users)
        self.ids_item = range(0, self.number_of_items)
        self.ids_cold_user = np.array(
            [user for user in range(0, self.number_of_users) if user not in self.ids_warm_user])
        self.ids_cold_item = np.array(
            [item for item in range(0, self.number_of_items) if item not in self.ids_warm_item])
        self.number_of_warm_users = self.ids_warm_user.shape[0]
        self.number_of_warm_items = self.ids_warm_item.shape[0]
        self.number_of_cold_users = self.ids_cold_user.shape[0]
        self.number_of_cold_items = self.ids_cold_item.shape[0]
        self.ids_target_users = np.array(data_reader.load_target())
        self.number_of_target_users = self.ids_target_users.shape[0]
        self.icm_asset = data_reader.load_icm_asset()
        self.icm_price = data_reader.load_icm_price()
        self.icm_class = data_reader.load_icm_class()
        self.ucm_region = data_reader.load_ucm_region(self.number_of_users)
        self.ucm_age = data_reader.load_ucm_age(self.number_of_users)
        self.ucm_all = sps.hstack([self.ucm_region, self.ucm_age])
        splitter = Splitter(self.urm)
        splitter.split_train_test_check_if_stored(k=k, probability=0, random_seed=random_seed)
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
        self.number_of_interactions_per_user = (self.urm > 0).sum(axis=1)
        self.number_of_interactions_per_item = (self.urm > 0).sum(axis=0)
        self.number_of_region_per_user = (self.ucm_region > 0).sum(axis=1)
        self.number_of_user_per_region = (self.ucm_region > 0).sum(axis=0)
        self.number_of_age_per_user = (self.ucm_age > 0).sum(axis=1)
        self.number_of_user_per_age = (self.ucm_age > 0).sum(axis=0)
        self.number_of_interactions_per_train_user = (self.urm_train > 0).sum(axis=1)
        self.number_of_interactions_per_train_item = (self.urm_train > 0).sum(axis=0)
        self.urm_users_by_type = np.array([(self.get_number_of_users_with_from_X_to_Y_interactions(0, 0),
                                            self.get_ids_of_users_with_from_X_to_Y_interactions(0, 0),
                                            "users with [0, 0] interactions"),
                                           (self.get_number_of_users_with_from_X_to_Y_interactions(1, 1),
                                            self.get_ids_of_users_with_from_X_to_Y_interactions(1, 1),
                                            "users with [1, 1] interactions"),
                                           (self.get_number_of_users_with_from_X_to_Y_interactions(2, 3),
                                            self.get_ids_of_users_with_from_X_to_Y_interactions(2, 3),
                                            "users with [2, 3] interactions"),
                                           (self.get_number_of_users_with_from_X_to_Y_interactions(4, 5),
                                            self.get_ids_of_users_with_from_X_to_Y_interactions(4, 5),
                                            "users with [4, 5] interactions"),
                                           (self.get_number_of_users_with_from_X_to_Y_interactions(6, 7),
                                            self.get_ids_of_users_with_from_X_to_Y_interactions(6, 7),
                                            "users with [6, 7] interactions"),
                                           (self.get_number_of_users_with_from_X_to_Y_interactions(8, 10),
                                            self.get_ids_of_users_with_from_X_to_Y_interactions(8, 10),
                                            "users with [8, 10] interactions"),
                                           (self.get_number_of_users_with_from_X_to_Y_interactions(11, 14),
                                            self.get_ids_of_users_with_from_X_to_Y_interactions(11, 14),
                                            "users with [11, 14] interactions"),
                                           (self.get_number_of_users_with_from_X_to_Y_interactions(15, 20),
                                            self.get_ids_of_users_with_from_X_to_Y_interactions(15, 20),
                                            "users with [15, 20] interactions"),
                                           (self.get_number_of_users_with_from_X_to_Y_interactions(21, 31),
                                            self.get_ids_of_users_with_from_X_to_Y_interactions(21, 31),
                                            "users with [21, 31] interactions"),
                                           (self.get_number_of_users_with_more_than_X_interactions(31),
                                            self.get_ids_of_users_with_more_than_X_interactions(31),
                                           "users with [31, ) interactions")])
        self.urm_train_users_by_type = np.array([(self.get_number_of_train_users_with_from_X_to_Y_interactions(0, 0),
                                                  self.get_ids_of_train_users_with_from_X_to_Y_interactions(0, 0),
                                                  "train users with [0, 0] interactions"),
                                                 (self.get_number_of_train_users_with_from_X_to_Y_interactions(1, 1),
                                                  self.get_ids_of_train_users_with_from_X_to_Y_interactions(1, 1),
                                                  "train users with [1, 1] interactions"),
                                                 (self.get_number_of_train_users_with_from_X_to_Y_interactions(2, 3),
                                                  self.get_ids_of_train_users_with_from_X_to_Y_interactions(2, 3),
                                                  "train users with [2, 3] interactions"),
                                                 (self.get_number_of_train_users_with_from_X_to_Y_interactions(4, 5),
                                                  self.get_ids_of_train_users_with_from_X_to_Y_interactions(4, 5),
                                                  "train users with [4, 5] interactions"),
                                                 (self.get_number_of_train_users_with_from_X_to_Y_interactions(6, 7),
                                                  self.get_ids_of_train_users_with_from_X_to_Y_interactions(6, 7),
                                                  "train users with [6, 7] interactions"),
                                                 (self.get_number_of_train_users_with_from_X_to_Y_interactions(8, 10),
                                                  self.get_ids_of_train_users_with_from_X_to_Y_interactions(8, 10),
                                                  "train users with [8, 10] interactions"),
                                                 (self.get_number_of_train_users_with_from_X_to_Y_interactions(11, 14),
                                                  self.get_ids_of_train_users_with_from_X_to_Y_interactions(11, 14),
                                                  "train users with [11, 14] interactions"),
                                                 (self.get_number_of_train_users_with_from_X_to_Y_interactions(15, 20),
                                                  self.get_ids_of_train_users_with_from_X_to_Y_interactions(15, 20),
                                                  "train users with [15, 20] interactions"),
                                                 (self.get_number_of_train_users_with_from_X_to_Y_interactions(21, 31),
                                                  self.get_ids_of_train_users_with_from_X_to_Y_interactions(21, 31),
                                                  "train users with [21, 31] interactions"),
                                                 (self.get_number_of_train_users_with_from_X_to_Y_interactions(31),
                                                  self.get_ids_of_train_users_with_more_than_X_interactions(31),
                                                 "train users with [31, ) interactions")])

    def clone(self):
        return copy.deepcopy(self)

    def print(self):
        print(f"urm size: {self.urm.shape}\n"
              f"urm interactions: {self.urm.nnz} [{round(self.urm.nnz / self.urm.nnz * 100, 2)}%]\n"
              f"number of users: {self.number_of_users} [{round(self.number_of_users / self.number_of_users * 100, 2)}%]\n"
              f"number of items: {self.number_of_items} [{round(self.number_of_items / self.number_of_items * 100, 2)}%]\n"
              f"number of interactions per user max: {self.number_of_interactions_per_user.max()}\n"
              f"number of interactions per item max: {self.number_of_interactions_per_item.max()}\n"
              f"number of interactions per user avg: {round(self.number_of_interactions_per_user.mean(), 2)}\n"
              f"number of interactions per item avg: {round(self.number_of_interactions_per_item.mean(), 2)}\n"
              f"number of interactions per train user max: {self.number_of_interactions_per_train_user.max()}\n"
              f"number of interactions per train item max: {self.number_of_interactions_per_train_item.max()}\n"
              f"number of interactions per train user avg: {round(self.number_of_interactions_per_train_user.mean(), 2)}\n"
              f"number of interactions per train item avg: {round(self.number_of_interactions_per_train_item.mean(), 2)}\n"
              f"number of region per user max: {self.number_of_region_per_user.max()}\n"
              f"number of user per region max: {self.number_of_user_per_region.max()}\n"
              f"number of region per user avg: {round(self.number_of_region_per_user.mean(), 2)}\n"
              f"number of user per region avg: {round(self.number_of_user_per_region.mean(), 2)}\n"
              f"number of age per user max: {self.number_of_age_per_user.max()}\n"
              f"number of user per age max: {self.number_of_user_per_age.max()}\n"
              f"number of age per user avg: {round(self.number_of_age_per_user.mean(), 2)}\n"
              f"number of user per age avg: {round(self.number_of_user_per_age.mean(), 2)}\n"
              f"number of warm users in urm: {self.number_of_warm_users} [{round(self.number_of_warm_users / self.number_of_users * 100, 2)}%]\n"
              f"number of warm items in urm: {self.number_of_warm_items} [{round(self.number_of_warm_items / self.number_of_items * 100, 2)}%]\n"
              f"number of cold users in urm: {self.number_of_cold_users} [{round(self.number_of_cold_users / self.number_of_users * 100, 2)}%]\n"
              f"number of cold items in urm: {self.number_of_cold_items} [{round(self.number_of_cold_items / self.number_of_items * 100, 2)}%]\n"
              f"train urm size: {self.urm_train.shape}\n"
              f"train urm interactions: {self.urm_train.nnz} [{round(self.urm_train.nnz / self.urm.nnz * 100, 2)}%]\n"
              f"number of warm users in train urm: {self.number_of_warm_train_users} [{round(self.number_of_warm_train_users / self.number_of_users * 100, 2)}%]\n"
              f"number of warm items in train urm: {self.number_of_warm_train_items} [{round(self.number_of_warm_train_items / self.number_of_items * 100, 2)}%]\n"
              f"number of cold users in train urm: {self.number_of_cold_train_users} [{round(self.number_of_cold_train_users / self.number_of_users * 100, 2)}%]\n"
              f"number of cold items in train urm: {self.number_of_cold_train_items} [{round(self.number_of_cold_train_items / self.number_of_items * 100, 2)}%]\n"
              f"test urm size: {self.urm_train.shape}\n"
              f"test urm interactions: {self.urm_test.nnz} [{round(self.urm_test.nnz / self.urm.nnz * 100, 2)}%]\n")

    def get_number_of_users_with_less_than_X_interactions(self, x=100):
        a = self.number_of_interactions_per_user
        a = np.where(a < x)
        return a[0].shape[0]

    def get_ids_of_users_with_less_than_X_interactions(self, x=100):
        a = self.number_of_interactions_per_user
        a = np.where(a < x)
        return a[0]

    def get_number_of_train_users_with_less_than_X_interactions(self, x=100):
        a = self.number_of_interactions_per_train_user
        a = np.where(a < x)
        return a[0].shape[0]

    def get_ids_of_train_users_with_less_than_X_interactions(self, x=100):
        a = self.number_of_interactions_per_train_user
        a = np.where(a < x)
        return a[0]

    def get_number_of_users_with_more_than_X_interactions(self, x=100):
        a = self.number_of_interactions_per_user
        a = np.where(a > x)
        return a[0].shape[0]

    def get_ids_of_users_with_more_than_X_interactions(self, x=100):
        a = self.number_of_interactions_per_user
        a = np.where(a > x)
        return a[0]

    def get_number_of_train_users_with_more_than_X_interactions(self, x=100):
        a = self.number_of_interactions_per_train_user
        a = np.where(a > x)
        return a[0].shape[0]

    def get_ids_of_train_users_with_more_than_X_interactions(self, x=100):
        a = self.number_of_interactions_per_train_user
        a = np.where(a > x)
        return a[0]

    def get_number_of_users_with_X_interactions(self, x=100):
        a = self.number_of_interactions_per_user
        a = np.where(a == x)
        return a[0].shape[0]

    def get_ids_of_users_with_X_interactions(self, x=100):
        a = self.number_of_interactions_per_user
        a = np.where(a == x)
        return a[0]

    def get_number_of_train_users_with_X_interactions(self, x=100):
        a = self.number_of_interactions_per_train_user
        a = np.where(a == x)
        return a[0].shape[0]

    def get_ids_of_train_users_with_X_interactions(self, x=100):
        a = self.number_of_interactions_per_train_user
        a = np.where(a == x)
        return a[0]

    def get_number_of_users_with_from_X_to_Y_interactions(self, x=100, y=200):
        a = self.number_of_interactions_per_user
        x1 = np.where(a >= x)
        x2 = np.where(a <= y)
        return np.array([x for x in x1[0] if x in x2[0]]).shape[0]

    def get_ids_of_users_with_from_X_to_Y_interactions(self, x=100, y=200):
        a = self.number_of_interactions_per_user
        x1 = np.where(a >= x)
        x2 = np.where(a <= y)
        return np.array([x for x in x1[0] if x in x2[0]])

    def get_number_of_train_users_with_from_X_to_Y_interactions(self, x=100, y=200):
        a = self.number_of_interactions_per_train_user
        x1 = np.where(a >= x)
        x2 = np.where(a <= y)
        return np.array([x for x in x1[0] if x in x2[0]]).shape[0]

    def get_ids_of_train_users_with_from_X_to_Y_interactions(self, x=100, y=200):
        a = self.number_of_interactions_per_train_user
        x1 = np.where(a >= x)
        x2 = np.where(a <= y)
        return np.array([x for x in x1[0] if x in x2[0]])
