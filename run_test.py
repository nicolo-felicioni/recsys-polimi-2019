from Base.Evaluation import MyEvaluator
from Base.Evaluation.Evaluator import EvaluatorHoldout
from DataObject import DataObject
from DataReader import DataReader
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
#from MatrixFactorization.WRMFRecommender import WRMFRecommender

import numpy as np
import pandas as pd
import scipy.sparse as sps

from Base.NonPersonalizedRecommender import TopPop, Random, GlobalEffects

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender

from Data_manager.RecSys2019.RecSys2019Reader import RecSys2019Reader
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out

import traceback, os


def load_target():
    df_original = pd.read_csv(filepath_or_buffer="Data_manager_split_datasets/RecSys2019/alg_sample_submission.csv", sep=',', header=0,
                              dtype={'user': int, 'items': str})

    df_original.columns = ['user', 'items']

    user_id_list = df_original['user'].values

    user_id_unique = np.unique(user_id_list)

    print("DataReader:")
    print("\tLoading the target users:")
    print("\t\tTarget size:" + str(user_id_unique.shape))
    print("\tTarget users loaded.")

    return user_id_unique

if __name__ == '__main__':

    output_root_path = "./result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)


    logFile = open(output_root_path + "result_all_algorithms.txt", "a")

    data_reader = DataReader()
    data = DataObject(data_reader, 1)
    # cold_recommender = TopPop(data.train_urm)
    recommender = MatrixFactorization_BPR_Cython(data.train_urm)
    # cold_recommender.fit()

    # for user_id in data.cold_user_ids:
    #     recommended_items = cold_recommender.recommend(user_id, cutoff=10)
    #     well_formatted = " ".join([str(x) for x in recommended_items])
    #     print(f"{user_id}, {well_formatted}\n")

    # for ep in [10, 100, 1000, 10000]:
    #     for l_i in [0.001, 0.01, 0.1]:
    #         for l_j in [0.001, 0.01, 0.1]:
    #             for l_rate in [0.00001, 0.0001, 0.001, 0.01]:
    #                 recommender.fit(epochs=ep, lambda_i=l_i, lambda_j=l_j, learning_rate=l_rate, random_seed=17)
    #                 evaluator = EvaluatorHoldout(data.test_urm, [10], exclude_seen=True)
    #                 results_run, results_run_string = evaluator.evaluateRecommender(recommender)
    #
    #
    #
    #                 print("Algorithm: {}, results: \n{}".format(recommender.__class__, results_run_string))
    #                 logFile.write(f"Epoch: {ep}\nLambda_i: {l_i}\nLambda_j: {l_j}\nLearning rate: {l_rate}\n")
    #                 logFile.write("Algorithm: {}, results: \n{}\n".format(recommender.__class__, results_run_string))
    #                 logFile.flush()

    recommender.fit()
    MyEvaluator.evaluate_algorithm(data.test_urm, data.warm_user_ids, recommender)

    # recommender.fit(epochs=1100)
    # f = open("submission.csv", "w+")
    # f.write("user_id,item_list\n")
    # for user_id in data.target_users:
    #     if len(data.train_urm[user_id].indices) > 1:
    #         recommended_items = recommender.recommend(user_id, cutoff=10)
    #     else:
    #         recommended_items = cold_recommender.recommend(user_id, cutoff=10)
    #     well_formatted = " ".join([str(x) for x in recommended_items])
    #     f.write(f"{user_id}, {well_formatted}\n")

    # dataset_object = RecSys2019Reader()
    #
    # dataSplitter = DataSplitter_leave_k_out(dataset_object)
    #
    # dataSplitter.load_data()
    # URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
    #
    # target_users = load_target()
    #
    # recommender_list = [
    #     SLIM_BPR_Cython
    # ]
    #
    #
    # from Base.Evaluation.Evaluator import EvaluatorHoldout
    #
    # evaluator = EvaluatorHoldout(URM_test, [10], exclude_seen=True)
    #
    #
    # output_root_path = "./result_experiments/"
    #
    # # If directory does not exist, create
    # if not os.path.exists(output_root_path):
    #     os.makedirs(output_root_path)
    #
    #
    # logFile = open(output_root_path + "result_all_algorithms.txt", "a")
    # f = open("submission.csv", "w+")
    #
    # for recommender_class in recommender_list:
    #
    #     try:
    #
    #         print("Algorithm: {}".format(recommender_class))
    #
    #
    #         recommender = recommender_class(URM_train)
    #         recommender.fit()
    #
    #         for user_id in target_users:
    #             recommended_items = recommender.recommend(user_id, cutoff=10)
    #             well_formatted = " ".join([str(x) for x in recommended_items])
    #             f.write(f"{user_id}, {well_formatted}\n")
    #
    #
    #     #     results_run, results_run_string = evaluator.evaluateRecommender(recommender)
    #     #     print("Algorithm: {}, results: \n{}".format(recommender.__class__, results_run_string))
    #     #     logFile.write("Algorithm: {}, results: \n{}\n".format(recommender.__class__, results_run_string))
    #     #     logFile.flush()
    #     #
    #     except Exception as e:
    #         traceback.print_exc()
    #         logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
    #         logFile.flush()

