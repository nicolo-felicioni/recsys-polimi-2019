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
        self.number_of_warm_items = self.ids_warm_item.shape[0]
        self.number_of_cold_users = len(self.ids_cold_user)
        self.number_of_cold_items = len(self.ids_cold_item)
        self.ids_target_users = data_reader.load_target()
        self.number_of_target_users = len(self.ids_target_users)
        self.icm_asset = data_reader.load_icm_asset()
        self.icm_price = data_reader.load_icm_price()
        self.icm_class = data_reader.load_icm_class()
        self.ucm_region = data_reader.load_ucm_region()
        splitter = Splitter(self.urm)
        splitter.split_train_test(k=0, probability=0, random_seed=17)
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
        self.number_of_interactions_per_user = (self.urm > 0).sum(axis=0)
        self.number_of_interactions_per_item = (self.urm > 0).sum(axis=1)

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