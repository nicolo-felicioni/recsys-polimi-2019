import time

from Base.Evaluation import MyEvaluator
from DataObject import DataObject
from DataReader import DataReader
from Hybrid.Hybrid001AlphaRecommender import Hybrid001AlphaRecommender
from Hybrid.Hybrid002AlphaRecommender import Hybrid002AlphaRecommender
from Hybrid.Hybrid003AlphaRecommender import Hybrid003AlphaRecommender
from KNN.UserKNNAgeRecommender import UserKNNAgeRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
import pandas as pd

from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
#from MatrixFactorization.WRMFRecommender import WRMFRecommender

from Base.NonPersonalizedRecommender import TopPop, Random, GlobalEffects

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender

from Data_manager.RecSys2019.RecSys2019Reader import RecSys2019Reader
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out

import traceback, os


def make_submission():
    print("making the submission")
    data_reader_loc = DataReader()
    data_loc = DataObject(data_reader_loc, k=0)
    recommender = Hybrid003AlphaRecommender(data_loc)
    recommender.fit()
    f = open("submissionHybrid003_k_12_shrink_15.csv", "w+")
    f.write("user_id,item_list\n")
    for user_id in data_loc.ids_target_users:
        recommended_items = recommender.recommend(user_id, cutoff=10)
        well_formatted = " ".join([str(x) for x in recommended_items])
        f.write(f"{user_id}, {well_formatted}\n")





if __name__ == '__main__':

    # df_user_age = pd.read_csv("Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_UCM_age.csv")


    want_submission = False
    if want_submission:
        make_submission()
    else:
        start_time = time.time()
        data_reader = DataReader()
        data = DataObject(data_reader, k=1, random_seed=15)
        stop_time = time.time()

        print("Completed in {:.2f} seconds".format(float(time.time() - start_time)))

        # rec1 = Hybrid003AlphaRecommender(data)
        # rec1.fit()
        # print(MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_user, rec1))
        #rec2 = Hybrid002AlphaRecommender(data.urm_train, data.ucm_region, data.ids_cold_train_users, data.ids_warm_train_users)
        #rec2.fit()
        #MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_cold_train_users, rec2)

    '''
    data_reader = DataReader()
    data = DataObject(data_reader, k=0)
    print(type(data.ids_target_users))
    recommender = Hybrid001AlphaRecommender(data.urm_train, data.ucm_region, data.ids_cold_train_users,
                                     data.ids_warm_train_users)
    recommender.fit()
    f = open("submissionHybrid001.csv", "w+")
    f.write("user_id,item_list\n")

    for user_id in data.ids_target_users:
        recommended_items = recommender.recommend(user_id, cutoff=10)
        well_formatted = " ".join([str(x) for x in recommended_items])
        f.write(f"{user_id}, {well_formatted}\n")

    '''


