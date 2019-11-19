#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import zipfile, shutil

from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL, merge_ICM
import pandas as pd


class RecSys2019Reader(DataReader):
    DATASET_URL = "https://www.kaggle.com/c/15854/download-all"
    DATASET_SUBFOLDER = "RecSys2019/"
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = ["ICM_classes"]
    # AVAILABLE_ICM = ["ICM_asset", "ICM_price", "ICM_sub_class"]
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = True

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):

        zipFile_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "recommender-system-2019-challenge-polimi.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("Movielens10MReader: Unable to fild data zip file. Downloading...")

            downloadFromURL(self.DATASET_URL, zipFile_path, "ml-10m.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-10m.zip")

        URM_path = dataFile.extract("data_train.csv", path=zipFile_path + "decompressed/")
        ICM_asset = dataFile.extract("data_ICM_asset.csv", path=zipFile_path + "decompressed/")
        ICM_price = dataFile.extract("data_ICM_price.csv", path=zipFile_path + "decompressed/")
        ICM_sub_class = dataFile.extract("data_ICM_sub_class.csv", path=zipFile_path + "decompressed/")

        # print("RecSys2019Reader: loading ICM assets")
        # ICM_assets, tokenToFeatureMapper_ICM_assets, self.item_original_ID_to_index = _loadICM_assets(ICM_asset,
        #                                                                                               header=True,
        #                                                                                               separator=',',
        #                                                                                               genresSeparator="|")
        # self._LOADED_ICM_DICT["ICM_assets"] = ICM_assets
        # self._LOADED_ICM_MAPPER_DICT["ICM_assets"] = tokenToFeatureMapper_ICM_assets
        #
        # print("RecSys2019Reader: loading ICM price")
        # ICM_prices, tokenToFeatureMapper_ICM_prices, self.item_original_ID_to_index = _loadICM_assets(ICM_price,
        #                                                                                               header=True,
        #                                                                                               separator=',',
        #                                                                                               genresSeparator="|")
        # self._LOADED_ICM_DICT["ICM_prices"] = ICM_assets
        # self._LOADED_ICM_MAPPER_DICT["ICM_prices"] = tokenToFeatureMapper_ICM_assets
        # print("Movielens10MReader: loading genres")
        # ICM_assets, tokenToFeatureMapper_ICM_assets, self.item_original_ID_to_index = _loadICM_genres(genres_path, header=True, separator='::', genresSeparator="|")
        #
        # self._LOADED_ICM_DICT["ICM_assets"] = ICM_assets
        # self._LOADED_ICM_MAPPER_DICT["ICM_assets"] = tokenToFeatureMapper_ICM_assets
        #
        # print("Movielens10MReader: loading tags")
        # ICM_prices, tokenToFeatureMapper_ICM_prices, _ = _loadICM_tags(tags_path, header=True, separator='::', if_new_item = "ignore",
        #                                                                      item_original_ID_to_index = self.item_original_ID_to_index)
        print("RecSys2019Reader: loading ICM class")
        ICM_classes, tokenToFeatureMapper_ICM_classes, self.item_original_ID_to_index = _loadICM_assets(ICM_sub_class,
                                                                                                      header=True,
                                                                                                      separator=',',
                                                                                                      genresSeparator="|")
        self._LOADED_ICM_DICT["ICM_classes"] = ICM_classes
        self._LOADED_ICM_MAPPER_DICT["ICM_classes"] = tokenToFeatureMapper_ICM_classes
        print("Movielens10MReader: loading genres")

        print("RecSys2019Reader: loading URM")
        URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = _loadURM_preinitialized_item_id(
            URM_path, separator=",",
            header=False, if_new_user="add", if_new_item="add",
            item_original_ID_to_index=self.item_original_ID_to_index)
        self._LOADED_URM_DICT["URM_all"] = URM_all
        self._LOADED_GLOBAL_MAPPER_DICT["user_original_ID_to_index"] = self.user_original_ID_to_index
        self._LOADED_GLOBAL_MAPPER_DICT["item_original_ID_to_index"] = self.item_original_ID_to_index

        print(URM_all.shape)

        # ICM_all, tokenToFeatureMapper_ICM_all = merge_ICM(ICM_assets, ICM_prices,
        #                                                   tokenToFeatureMapper_ICM_assets,
        #                                                   tokenToFeatureMapper_ICM_prices)
        #
        # self._LOADED_ICM_DICT["ICM_all"] = ICM_all
        # self._LOADED_ICM_MAPPER_DICT["ICM_all"] = tokenToFeatureMapper_ICM_all

        print("RecSys2019Reader: cleaning temporary files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("RecSys2019Reader: loading complete")


def _loadICM_assets(path, header=True, separator=',', genresSeparator="|"):
    # Genres
    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=None, on_new_col="add",
                                                    preinitialized_row_mapper=None, on_new_row="add")

    fileHandle = open(path, "r", encoding="latin1")
    numCells = 0

    if header:
        fileHandle.readline()

    for line in fileHandle:
        numCells += 1
        if (numCells % 10000 == 0):
            print("Processed {} cells".format(numCells))

        if (len(line)) > 1:
            line = line.split(separator)

            line[-1] = line[-1].replace("\n", "")

            movie_id = line[0]
            # In case the title contains commas, it is enclosed in "..."
            # genre list will always be the last element
            assetList = line[1]

            assetList = assetList.split(genresSeparator)
            data = line[2]

            # Rows movie ID
            # Cols features
            ICM_builder.add_single_row(movie_id, assetList, data=data)

    fileHandle.close()

    return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()


def _loadURM_preinitialized_item_id(URM_path, header=False, separator=",",
                                    if_new_user="add", if_new_item="add",
                                    item_original_ID_to_index=None,
                                    user_original_ID_to_index=None):
    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    URM_all_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=item_original_ID_to_index,
                                                        on_new_col=if_new_item,
                                                        preinitialized_row_mapper=user_original_ID_to_index,
                                                        on_new_row=if_new_user)

    if header:
        df_original = pd.read_csv(filepath_or_buffer=URM_path, sep=separator, header=0 if header else None,
                                  usecols=['user', 'item', 'rating'],
                                  dtype={'user': str, 'item': str, 'rating': float})
    else:
        df_original = pd.read_csv(filepath_or_buffer=URM_path, sep=separator, header=0 if header else None,
                                  dtype={0: str, 1: str, 2: float})

        df_original.columns = ['user', 'item', 'rating']

    # Remove data with rating non valid
    # df_original.drop(df_original[df_original.rating == 0.0].index, inplace=True)

    user_id_list = df_original['user'].values
    item_id_list = df_original['item'].values
    rating_list = df_original['rating'].values

    print(user_id_list.shape)
    print(len(set(item_id_list)))
    print(rating_list.shape)

    URM_all_builder.add_data_lists(user_id_list, item_id_list, rating_list)

    return URM_all_builder.get_SparseMatrix(), \
           URM_all_builder.get_column_token_to_id_mapper(), \
           URM_all_builder.get_row_token_to_id_mapper()
