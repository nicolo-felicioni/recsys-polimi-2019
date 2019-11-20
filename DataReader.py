import pandas as pd
import scipy.sparse as sps
import numpy as np

urm_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_train.csv"
icm_asset_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_ICM_asset.csv"
icm_price_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_ICM_price.csv"
icm_class_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_ICM_sub_class.csv"
target_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/alg_sample_submission.csv"


class DataReader(object):

    def load_urm(self):
        df_original = pd.read_csv(filepath_or_buffer=urm_path, sep=',', header=0,
                                  dtype={'row': int, 'col': int, 'rating': float})

        df_original.columns = ['user', 'item', 'rating']

        user_id_list = df_original['user'].values
        item_id_list = df_original['item'].values
        rating_id_list = df_original['rating'].values

        user_id_unique = np.unique(user_id_list)
        item_id_unique = np.unique(item_id_list)

        csr_matrix = sps.csr_matrix((rating_id_list, (user_id_list, item_id_list)))

        print("DataReader:")
        print("\tLoading the URM:")
        print("\t\tURM size:" + str(csr_matrix.shape))
        print("\t\tURM unique users:" + str(user_id_unique.size))
        print("\t\tURM unique items:" + str(item_id_unique.size))
        print("\tURM loaded.")

        return csr_matrix, user_id_unique, item_id_unique

    def load_target(self):
        df_original = pd.read_csv(filepath_or_buffer=target_path, sep=',', header=0,
                                  dtype={'user': int, 'items': str})

        df_original.columns = ['user', 'items']

        user_id_list = df_original['user'].values

        user_id_unique = np.unique(user_id_list)

        print("DataReader:")
        print("\tLoading the target users:")
        print("\t\tTarget size:" + str(user_id_unique.shape))
        print("\tTarget users loaded.")

        return user_id_unique

    def load_icm_asset(self):
        df_original = pd.read_csv(filepath_or_buffer=icm_asset_path, sep=',', header=1,
                                  dtype={'item': int, 'feature': int, 'data': float})

        df_original.columns = ['item', 'feature', 'data']

        item_id_list = df_original['item'].values
        feature_id_list = df_original['feature'].values
        data_id_list = df_original['data'].values

        csr_matrix = sps.csr_matrix((data_id_list, (item_id_list, feature_id_list)))

        print("DataReader:")
        print("\tLoading the asset ICM: " + icm_asset_path)
        print("\t\tAsset ICM size:" + str(csr_matrix.shape))
        print("\tAsset ICM loaded.")

        return csr_matrix

    def load_icm_price(self):
        df_original = pd.read_csv(filepath_or_buffer=icm_price_path, sep=',', header=1,
                                  dtype={'item': int, 'feature': int, 'data': float})

        df_original.columns = ['item', 'feature', 'data']

        item_id_list = df_original['item'].values
        feature_id_list = df_original['feature'].values
        data_id_list = df_original['data'].values

        csr_matrix = sps.csr_matrix((data_id_list, (item_id_list, feature_id_list)))

        print("DataReader:")
        print("\tLoading the price ICM: " + icm_asset_path)
        print("\t\tPrice ICM size:" + str(csr_matrix.shape))
        print("\tPrice ICM loaded.")

        return csr_matrix

    def load_icm_class(self):
        df_original = pd.read_csv(filepath_or_buffer=icm_class_path, sep=',', header=1,
                                  dtype={'item': int, 'feature': int, 'data': float})

        df_original.columns = ['item', 'feature', 'data']

        item_id_list = df_original['item'].values
        feature_id_list = df_original['feature'].values
        data_id_list = df_original['data'].values

        csr_matrix = sps.csr_matrix((data_id_list, (item_id_list, feature_id_list)))

        print("DataReader:")
        print("\tLoading the class ICM: " + icm_asset_path)
        print("\t\tClass ICM size:" + str(csr_matrix.shape))
        print("\tClass ICM loaded.")

        return csr_matrix

