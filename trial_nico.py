from Base.Evaluation import MyEvaluator
from DataObject import DataObject
from DataReader import DataReader
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


if __name__ == '__main__':

    df_user_age = pd.read_csv("Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_UCM_age.csv")
    data_reader = DataReader()
    data = DataObject(data_reader)
    # print(data.urm_train.shape)

    # rec = UserKNNAgeRecommender(df_user_age, data.urm_train)
    # rec = TopPop(data.urm_train)



    rec.fit(shrink=0, topK=50)
    MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_cold_train_users, rec)
