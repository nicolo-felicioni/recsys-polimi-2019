import pandas as pd
import scipy.sparse as sps
import numpy as np

urm_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_train.csv"
icm_asset_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_ICM_asset.csv"
icm_price_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_ICM_price.csv"
icm_class_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_ICM_sub_class.csv"
ucm_region_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_UCM_region.csv"
ucm_age_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_UCM_age.csv"
target_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/alg_sample_submission.csv"
ucm_interaction_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_UCM_interaction.csv"

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
        csr_matrix = csr_matrix.astype(dtype=np.float)
        # print("DataReader:")
        # print("\tLoading the URM:")
        # print("\t\tURM size:" + str(csr_matrix.shape))
        # print("\t\tURM unique users:" + str(user_id_unique.size))
        # print("\t\tURM unique items:" + str(item_id_unique.size))
        # print("\tURM loaded.")

        return csr_matrix, user_id_unique, item_id_unique

    def load_target(self):
        df_original = pd.read_csv(filepath_or_buffer=target_path, sep=',', header=0,
                                  dtype={'user': int, 'items': str})

        df_original.columns = ['user', 'items']

        user_id_list = df_original['user'].values

        user_id_unique = np.unique(user_id_list)

        # print("DataReader:")
        # print("\tLoading the target users:")
        # print("\t\tTarget size:" + str(user_id_unique.shape))
        # print("\tTarget users loaded.")

        return user_id_unique

    def load_icm_asset(self):
        df_original = pd.read_csv(filepath_or_buffer=icm_asset_path, sep=',', header=0,
                                  dtype={'item': int, 'feature': int, 'data': float})

        df_original.columns = ['item', 'feature', 'data']

        item_id_list = df_original['item'].values
        feature_id_list = df_original['feature'].values
        data_id_list = df_original['data'].values * 2

        csr_matrix = sps.csr_matrix((data_id_list, (item_id_list, feature_id_list)))

        # print("DataReader:")
        # print("\tLoading the asset ICM: " + icm_asset_path)
        # print("\t\tAsset ICM size:" + str(csr_matrix.shape))
        # print("\tAsset ICM loaded.")

        return csr_matrix

    def load_icm_price(self):
        df_original = pd.read_csv(filepath_or_buffer=icm_price_path, sep=',', header=0,
                                  dtype={'item': int, 'feature': int, 'data': float})

        df_original.columns = ['item', 'feature', 'data']

        item_id_list = df_original['item'].values
        feature_id_list = [x+1 for x in df_original['feature'].values]
        data_id_list = df_original['data'].values * 2

        csr_matrix = sps.csr_matrix((data_id_list, (item_id_list, feature_id_list)))

        # print("DataReader:")
        # print("\tLoading the price ICM: " + icm_asset_path)
        # print("\t\tPrice ICM size:" + str(csr_matrix.shape))
        # print("\tPrice ICM loaded.")

        return csr_matrix

    def load_icm_asset_augmented(self):
        df_original = pd.read_csv(filepath_or_buffer=icm_asset_path, sep=',', header=0,
                                  dtype={'item': int, 'feature': int, 'data': float})

        df_original.columns = ['item', 'feature', 'data']

        item_id_list = df_original['item'].values
        feature_id_list = [x+1 for x in df_original['feature'].values]
        data_id_list = df_original['data'].values

        all_list = zip(item_id_list, feature_id_list, data_id_list)
        sorted_list = sorted(all_list, key=lambda v: v[2])
        n_cluster = 5
        off_set = 0
        size = len(sorted_list)

        new_item_id_list = []
        new_feature_id_list = []
        new_data_id_list = []

        for i in range(0,n_cluster):
            min = int(i * size/n_cluster)
            max = int((i+1) * size/n_cluster)
            for item, feature_id_list, data in sorted_list[min:max]:
                new_item_id_list.append(int(item))
                new_feature_id_list.append(int(i) + off_set)
                new_data_id_list.append(0.5)
                new_item_id_list.append(int(item))
                new_feature_id_list.append(int(i+1) + off_set)
                new_data_id_list.append(1)
                new_item_id_list.append(int(item))
                new_feature_id_list.append(int(i+2) + off_set)
                new_data_id_list.append(0.5)


        csr_matrix = sps.csr_matrix((new_data_id_list, (new_item_id_list, new_feature_id_list)))

        # print("DataReader:")
        # print("\tLoading the price ICM: " + icm_asset_path)
        # print("\t\tPrice ICM size:" + str(csr_matrix.shape))
        # print("\tPrice ICM loaded.")

        return csr_matrix

    def load_icm_price_augmented(self):
        df_original = pd.read_csv(filepath_or_buffer=icm_price_path, sep=',', header=0,
                                  dtype={'item': int, 'feature': int, 'data': float})

        df_original.columns = ['item', 'feature', 'data']

        item_id_list = df_original['item'].values
        feature_id_list = [x+1 for x in df_original['feature'].values]
        data_id_list = df_original['data'].values

        all_list = zip(item_id_list, feature_id_list, data_id_list)
        sorted_list = sorted(all_list, key=lambda v: v[2])
        n_cluster = 5
        off_set = 8
        size = len(sorted_list)

        new_item_id_list = []
        new_feature_id_list = []
        new_data_id_list = []

        for i in range(0,n_cluster):
            min = int(i * size/n_cluster)
            max = int((i+1) * size/n_cluster)
            for item, feature_id_list, data in sorted_list[min:max]:
                new_item_id_list.append(int(item))
                new_feature_id_list.append(int(i) + off_set)
                new_data_id_list.append(0.5)
                new_item_id_list.append(int(item))
                new_feature_id_list.append(int(i+1) + off_set)
                new_data_id_list.append(1)
                new_item_id_list.append(int(item))
                new_feature_id_list.append(int(i+2) + off_set)
                new_data_id_list.append(0.5)


        csr_matrix = sps.csr_matrix((new_data_id_list, (new_item_id_list, new_feature_id_list)))

        # print("DataReader:")
        # print("\tLoading the price ICM: " + icm_asset_path)
        # print("\t\tPrice ICM size:" + str(csr_matrix.shape))
        # print("\tPrice ICM loaded.")

        return csr_matrix

    def load_icm_class(self):
        df_original = pd.read_csv(filepath_or_buffer=icm_class_path, sep=',', header=0,
                                  dtype={'item': int, 'feature': int, 'data': float})

        df_original.columns = ['item', 'feature', 'data']

        item_id_list = df_original['item'].values
        feature_id_list = np.array([x+2 for x in df_original['feature'].values])
        data_id_list = df_original['data'].values

        csr_matrix = sps.csr_matrix((data_id_list, (item_id_list, feature_id_list)))

        # print("DataReader:")
        # print("\tLoading the class ICM: " + icm_asset_path)
        # print("\t\tClass ICM size:" + str(csr_matrix.shape))
        # print("\tClass ICM loaded.")

        return csr_matrix

    def load_ucm_region(self, n):
        df_original = pd.read_csv(filepath_or_buffer=ucm_region_path, sep=',', header=0,
                                  dtype={'item': int, 'feature': int, 'data': float})

        df_original.columns = ['item', 'feature', 'data']

        item_id_list = df_original['item'].values
        feature_id_list = df_original['feature'].values
        data_id_list = df_original['data'].values

        csr_matrix = sps.csr_matrix((data_id_list, (item_id_list, feature_id_list)), shape=[n, max(feature_id_list)+1])

        # print("DataReader:")
        # print("\tLoading the region UCM: " + icm_asset_path)
        # print("\t\tRegion UCM size:" + str(csr_matrix.shape))
        # print("\tRegion UCM loaded.")

        return csr_matrix


    def load_ucm_interaction(self, n):
        df_original = pd.read_csv(filepath_or_buffer=ucm_interaction_path, sep=',', header=1,
                                  dtype={'item': int, 'feature': int, 'data': float})
        df_original.columns = ['item', 'feature', 'data']
        item_id_list = df_original['item'].values
        feature_id_list = df_original['feature'].values
        data_id_list = df_original['data'].values

        csr_matrix = sps.csr_matrix((data_id_list, (item_id_list, feature_id_list)), shape=[n, max(feature_id_list)+1])

        # print("DataReader:")
        # print("\tLoading the region UCM: " + icm_asset_path)
        # print("\t\tRegion UCM size:" + str(csr_matrix.shape))
        # print("\tRegion UCM loaded.")

        return csr_matrix

    def load_ucm_age(self, n):
        df_original = pd.read_csv(filepath_or_buffer=ucm_age_path, sep=',', header=0,
                                  dtype={'item': int, 'feature': int, 'data': float})

        df_original.columns = ['item', 'feature', 'data']

        item_id_list = df_original['item'].values
        feature_id_list = [x for x in df_original['feature'].values]
        data_id_list = df_original['data'].values

        csr_matrix = sps.csr_matrix((data_id_list, (item_id_list, feature_id_list)), shape=[n, max(feature_id_list)+1])

        # print("DataReader:")
        # print("\tLoading the age UCM: " + icm_asset_path)
        # print("\t\tAge UCM size:" + str(csr_matrix.shape))
        # print("\tAge UCM loaded.")

        return csr_matrix

