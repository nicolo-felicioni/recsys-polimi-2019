from Base.Evaluation import MyEvaluator
from Base.Evaluation.Evaluator import EvaluatorHoldout
from DataObject import DataObject
from DataReader import DataReader
from Hybrid.Hybrid000AlphaRecommender import Hybrid000AlphaRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
# from MatrixFactorization.WRMFRecommender import WRMFRecommender

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
    df_original = pd.read_csv(filepath_or_buffer="Data_manager_split_datasets/RecSys2019/alg_sample_submission.csv",
                              sep=',', header=0,
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
    data.print()


    # cold_recommender = TopPop(data.train_urm)
    # recommender = Hybrid000AlphaRecommender(data.urm_train, data.ucm_all, data.ids_cold_user, data.ids_warm_user)
    # recommender.fit(epochs=800, shrink=1, topK=15000)

    # for user_id in data.ids_warm_user:
    #     recommended_items = recommender.recommend(user_id, cutoff=10)
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

    # for e in [100, 200, 500, 1000, 1500, 2000]:
    #     for l_i in [1, 0.01, 0.0001]:
    #         for l_j in [1, 0.01, 0.0001]:
    #             for shrink in [0, 1, 2, 5, 10, 100]:
    #                 for k in [10, 100, 1000, 5000, 10000, 15000, 20000, 25000]:
    #                     recommender.fit(epochs=e, shrink=shrink, topK=k, lambda_j=l_j, lambda_i=l_i)
    #                     eval = MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_target_users, recommender)
    #                     print(eval)
    #                     f = open("eval.csv", "a+")
    #                     f.write(f"e: {e}\nl_i: {l_i}\nl_j: {l_j}\nshrink: {shrink}\nk: {k}")
    #                     f.write("a+")

    # f = open("eval.csv", "w+")
    # f.write("type_user, epoch, lambda_i, lambda_j, validation_every_n, lower_validation_allowed, map\n")
    # evaluator = EvaluatorHoldout(data.urm_test, [10], exclude_seen=True)
    # for e in [100, 500, 1000, 1500, 2000]:
    #     for l_i in [1, 0.01, 0.0001]:
    #         for l_j in [1, 0.01, 0.0001]:
    #             for n_e in [50, 100]:
    #                 for l_v_a in [1, 2, 5]:
    #                     print(f"{e}, {l_i}, {l_j}, {e/n_e}, {l_v_a}\n")
    #                     recommender = SLIM_BPR_Cython(data.urm_train)
    #                     recommender.fit(epochs=e, lambda_j=l_j, lambda_i=l_i,
    #                                     epochs_min=5,
    #                                     evaluator_object=evaluator,
    #                                     stop_on_validation=True,
    #                                     validation_every_n=n_e,
    #                                     validation_metric="MAP",
    #                                     lower_validations_allowed=l_v_a, random_seed=17)
    #                     result_string, map = MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_target_users, recommender)
    #                     print(result_string)
    #                     f.write(f"target, {e}, {l_i}, {l_j}, {n_e}, {l_v_a}, {map}\n")
    #                     f.flush()
    #                     result_string, map = MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_warm_user, recommender)
    #                     print(result_string)
    #                     f.write(f"warm, {e}, {l_i}, {l_j}, {n_e}, {l_v_a}, {map}\n")
    #                     f.flush()

    f = open("eval_user_cbf.csv", "w+")
    f.write("type_user, shrink, topk, feature_weight, ucm_type, map\n")
    evaluator = EvaluatorHoldout(data.urm_test, [10], exclude_seen=True)
    for shrink in [1, 2, 3, 4, 5, 10, 15, 20, 50, 100]:
        for k in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 11000, 12000, 13000, 14000, 15000]:
            for simil in ["BM25", "none", "TF-IDF"]:
                for type_ucm in [data.ucm_all, data.ucm_age, data.ucm_region]:
                    recommender = UserKNNCBFRecommender(type_ucm, data.urm_train)
                    recommender.fit(shrink=shrink, topK=k, feature_weighting=simil)
                    result_string, map = MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_target_users, recommender)
                    print(result_string)
                    if type_ucm is data.ucm_all:
                        f.write(f"target, {shrink}, {k}, {simil}, all, {map}\n")
                    else:
                        if type_ucm is data.ucm_age:
                            f.write(f"target, {shrink}, {k}, {simil}, age, {map}\n")
                        else:
                            f.write(f"target, {shrink}, {k}, {simil}, region, {map}\n")

                    f.flush()
                    result_string, map = MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_warm_train_users, recommender)
                    print(result_string)
                    if type_ucm is data.ucm_all:
                        f.write(f"warm, {shrink}, {k}, {simil}, all, {map}\n")
                    else:
                        if type_ucm is data.ucm_age:
                            f.write(f"warm, {shrink}, {k}, {simil}, age, {map}\n")
                        else:
                            f.write(f"warm, {shrink}, {k}, {simil}, region, {map}\n")
                    f.flush()
                    result_string, map = MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_cold_train_users, recommender)
                    print(result_string)
                    if type_ucm is data.ucm_all:
                        f.write(f"cold, {shrink}, {k}, {simil}, all, {map}\n")
                    else:
                        if type_ucm is data.ucm_age:
                            f.write(f"cold, {shrink}, {k}, {simil}, age, {map}\n")
                        else:
                            f.write(f"cold, {shrink}, {k}, {simil}, region, {map}\n")
                    f.flush()

    # f = open("submission.csv", "w+")
    # f.write("user_id,item_list\n")
    # for user_id in data.ids_target_users:
    #     recommended_items = recommender.recommend(user_id, cutoff=10)
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