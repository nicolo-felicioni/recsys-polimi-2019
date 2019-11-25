import operator
import copy
import math
import time

from joblib import Parallel, delayed

from Base.Evaluation import MyEvaluator, LogToFileEvaluator
from Base.Evaluation.Evaluator import EvaluatorHoldout
from DataObject import DataObject
from DataReader import DataReader
from Hybrid.Hybrid000AlphaRecommender import Hybrid000AlphaRecommender
from Hybrid.Hybrid001AlphaRecommender import Hybrid001AlphaRecommender
from Hybrid.Hybrid003AlphaRecommender import Hybrid003AlphaRecommender
from Hybrid.Hybrid100AlphaRecommender import Hybrid100AlphaRecommender
from Hybrid.Hybrid1XXAlphaRecommender import Hybrid1XXAlphaRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from MatrixFactorization.IALSRecommender import IALSRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import random as rnd

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


def mutate_weights(weights, mutation_factor=0.4, epoch=1, min_mutation=0.005, big_mutation_probability=0.005,
                   big_mutation_factor=10):
    weights = copy.deepcopy(weights)
    factor = mutation_factor / math.sqrt(epoch) + min_mutation
    partial_sum = 0
    for row in range(len(weights)):
        for col in range(len(weights[row])):
            if rnd.random() < big_mutation_probability:
                if rnd.random() < 0.5:
                    weights[row][col] = weights[row][col] / big_mutation_factor
                else:
                    weights[row][col] = weights[row][col] * big_mutation_factor
            else:
                weights[row][col] = weights[row][col] * ((1 - factor) + rnd.random() * factor)
            partial_sum = partial_sum + weights[row][col]

    for row in range(len(weights)):
        for col in range(len(weights[row])):
            weights[row][col] = weights[row][col] / partial_sum
    return weights


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


def make_submission():
    data_reader = DataReader()
    data = DataObject(data_reader, k=0)
    recommender = Hybrid003AlphaRecommender(data)
    recommender.fit()
    f = open("submission.csv", "w+")
    f.write("user_id,item_list\n")
    for user_id in data.ids_target_users:
        recommended_items = recommender.recommend(user_id, cutoff=10)
        well_formatted = " ".join([str(x) for x in recommended_items])
        f.write(f"{user_id}, {well_formatted}\n")


if __name__ == '__main__':
    want_submission = False

    if want_submission:
        make_submission()
    else:
        output_root_path = "./result_experiments/"

        # If directory does not exist, create
        if not os.path.exists(output_root_path):
            os.makedirs(output_root_path)

        logFile = open(output_root_path + "result_all_algorithms.txt", "a")

        random_seed = 15

        data_reader = DataReader()
        data = DataObject(data_reader, 1, random_seed=random_seed)
        data.print()

        # recommender = TopPop(data.urm_train)
        # recommender.fit()
        # users = data.ids_warm_train_users
        # eval, map = MyEvaluator.evaluate_algorithm_parallel(data.urm_test, users, recommender, at=10, remove_top=0)
        # print(f"FIRST 10,\t {description},\t {eval}")
        # IALSRecommender
        # recommender = ItemKNNCFRecommender(data.urm_train)
        # print(recommender)
        # shrink = 16
        # k = 12
        # fw = "none"
        # shrink = 20000
        # k = 20000
        # fw = "TF-IDF"
        # print(f"shrink {shrink}, k {k}, fw {fw}")
        # recommender.fit(shrink=shrink, topK=k, feature_weighting=fw)
        # recommender = SLIM_BPR_Cython(data.urm_train)
        # print(recommender)
        # recommender.fit(epochs=10000, learning_rate=0.000001, lambda_i=0.1, lambda_j=0.1)
        # recommender = UserKNNCBFRecommender(data.ucm_all, data.urm_train)
        # print(recommender)
        # recommender.fit(topK=12000, shrink=1)
        # recommender = Hybrid003AlphaRecommender(data)
        # recommender.fit()

        # for n, users, description in data.urm_train_users_by_type:
        #     print(f"{n} {users} {description}")

        # recommender = PureSVDRecommender(data)
        # recommender.fit(num_factors=5)
        # for n, users, description in data.urm_train_users_by_type:
        #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=5, remove_top=0)
        #     print(f"FIRST 5,\t {description},\t {eval}")
        # for n, users, description in data.urm_train_users_by_type:
        #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=5, remove_top=5)
        #     print(f"LAST 5,\t {description},\t {eval}")
        # for n, users, description in data.urm_train_users_by_type:
        #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
        #     print(f"ALL 10,\t {description},\t {eval}")
        # eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_target_users, recommender, at=10,
        #                                            remove_top=0)


        # print(f"TOTAL,\t \t {eval}")
        recommender = PureSVDRecommender(data.urm_train)
        recommender.fit(num_factors=5)

        LogToFileEvaluator.evaluate(data, random_seed, recommender, "PureSVD", "num_factors=5")
        # for n, users, description in data.urm_train_users_by_type:
        #     eval, map = MyEvaluator.evaluate_algorithm_parallel(data.urm_test, users, recommender, at=10, remove_top=0, parallelism=2)
        #     print(f"ALL 10,\t {description},\t {eval}")
        # eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_target_users, recommender, at=10,
        #                                            remove_top=0)
        # print(f"TOTAL,\t \t {eval}")



        # RECOMMENDER CHALLENGE
        # rec1 = ItemKNNCFRecommender(data.urm_train)
        # rec1.fit(shrink=20000, topK=20000, feature_weighting="TF-IDF")
        # rec2 = P3alphaRecommender(data.urm_train)
        # rec2.fit(topK=1000, implicit=True, alpha=0.4)
        # rec1 = TopPop(data.urm_train)
        # rec1.fit()
        # rec2 = UserKNNCBFRecommender(data.ucm_all, data.urm_train)
        # rec2.fit(topK=11000, shrink=1)
        # rec2 = SLIM_BPR_Cython(data.urm_train)
        # rec2.fit(epochs=1000, lambda_i=0.1, lambda_j=0.1)
        # for n, users, description in data.urm_train_users_by_type:
        #     size = 0
        #     for user_id in users:
        #         recommended_items1 = rec1.recommend(user_id, cutoff=10)
        #         recommended_items2 = rec2.recommend(user_id, cutoff=10)
        #         items = np.concatenate((recommended_items1, recommended_items2))
        #         size = size + np.unique(items).shape[0]
        #     average_size = size/n
        #
        #     print(f"ALL 10,\t {description},\t {average_size}")

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

        # f = open("eval_user_cbf.csv", "w+")
        # f.write("type_user, shrink, topk, feature_weight, ucm_type, map\n")
        # evaluator = EvaluatorHoldout(data.urm_test, [10], exclude_seen=True)
        # for shrink in [1, 2, 3, 4, 5, 10, 15, 20, 50, 100]:
        #     for k in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 11000, 12000, 13000, 14000, 15000]:
        #         for simil in ["BM25", "none", "TF-IDF"]:
        #             for type_ucm in [data.ucm_all, data.ucm_age, data.ucm_region]:
        #                 recommender = UserKNNCBFRecommender(type_ucm, data.urm_train)
        #                 recommender.fit(shrink=shrink, topK=k, feature_weighting=simil)
        #                 result_string, map = MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_target_users, recommender)
        #                 print(result_string)
        #                 if type_ucm is data.ucm_all:
        #                     f.write(f"target, {shrink}, {k}, {simil}, all, {map}\n")
        #                 else:
        #                     if type_ucm is data.ucm_age:
        #                         f.write(f"target, {shrink}, {k}, {simil}, age, {map}\n")
        #                     else:
        #                         f.write(f"target, {shrink}, {k}, {simil}, region, {map}\n")
        #
        #                 f.flush()
        #                 result_string, map = MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_warm_train_users, recommender)
        #                 print(result_string)
        #                 if type_ucm is data.ucm_all:
        #                     f.write(f"warm, {shrink}, {k}, {simil}, all, {map}\n")
        #                 else:
        #                     if type_ucm is data.ucm_age:
        #                         f.write(f"warm, {shrink}, {k}, {simil}, age, {map}\n")
        #                     else:
        #                         f.write(f"warm, {shrink}, {k}, {simil}, region, {map}\n")
        #                 f.flush()
        #                 result_string, map = MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_cold_train_users, recommender)
        #                 print(result_string)
        #                 if type_ucm is data.ucm_all:
        #                     f.write(f"cold, {shrink}, {k}, {simil}, all, {map}\n")
        #                 else:
        #                     if type_ucm is data.ucm_age:
        #                         f.write(f"cold, {shrink}, {k}, {simil}, age, {map}\n")
        #                     else:
        #                         f.write(f"cold, {shrink}, {k}, {simil}, region, {map}\n")
        #                 f.flush()

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

        # f = open("generation_log.csv", "w+")
        #
        # population_size = 6
        # epochs = 3
        # score = [None] * population_size
        # anti_overfitting = 2
        #
        # # recommenders[population_id][anti_overfitting_generation]
        # recommenders = np.full((population_size, anti_overfitting), None)
        # # base_recommenders[anti_overfitting_generation]
        # base_recommenders = []
        # tested_recommenders = []
        # datas = [None] * anti_overfitting
        #
        # for j in range(anti_overfitting):
        #     data_reader = DataReader()
        #     datas[j] = DataObject(data_reader, 1, random_seed=(50 + j*10))
        #     rec1 = ItemKNNCFRecommender(datas[j].urm_train)
        #     rec1.fit(shrink=15, topK=12, feature_weighting="none")
        #     rec2 = P3alphaRecommender(datas[j].urm_train)
        #     rec2.fit(topK=170, implicit=True, alpha=0.5)
        #     rec3 = PureSVDRecommender(datas[j].urm_train)
        #     rec3.fit(num_factors=20)
        #     base_recommenders.append([rec1, rec2, rec3])
        #
        # for i in range(population_size):
        #     weights = []
        #     weights.append([x / 15 for x in range(1, 20)][::-1])
        #     weights.append([x / 15 for x in range(1, 20)][::-1])
        #     weights.append([x / 15 for x in range(1, 20)][::-1])
        #     for j in range(anti_overfitting):
        #         rec = Hybrid1XXAlphaRecommender(datas[j], base_recommenders[j])
        #         rec.fit(weights=weights)
        #         recommenders[i][j] = rec
        #
        # for i in range(population_size):
        #     ws = mutate_weights(recommenders[i][0].weights)
        #     for j in range(anti_overfitting):
        #         recommenders[i][j].fit(weights=ws)
        #
        # start_time = time.time()
        #
        # for epoch in range(epochs):
        #     print(f"epoch {epoch}")
        #     f.write(f"epoch {epoch}\n")
        #     print("Epoch {} of {} complete in {:.2f} minutes".format(epoch, epochs,
        #                                                              float(time.time() - start_time) / 60))
        #     f.write("Epoch {} of {} complete in {:.2f} minutes".format(epoch, epochs,
        #                                                              float(time.time() - start_time) / 60))
        #     start_time = time.time()
        #
        #     for i in range(population_size):
        #         # partial_score = 0
        #         def parallel_run(j):
        #             _rec = recommenders[i][j]
        #             _n, _tested_users, _description = datas[j].urm_train_users_by_type[9]
        #             _result_string, _map = MyEvaluator.evaluate_algorithm_parallel(datas[j].urm_test, _tested_users, _rec)
        #             print(f"\t\t{j} map : {_map}")
        #             f.write(f"\t\t{j} map : {_map}\n")
        #             return _map
        #         partial_scores = Parallel(n_jobs=anti_overfitting)(delayed(parallel_run)(generation) for generation in range(anti_overfitting))
        #         average_score = 0
        #         for s in partial_scores:
        #             average_score += s
        #         average_score = average_score / anti_overfitting
        #
        #
        #
        #         # for j in range(anti_overfitting):
        #         #     rec = recommenders[i][j]
        #         #     n, tested_users, description = datas[j].urm_train_users_by_type[2]
        #         #     result_string, map = MyEvaluator.evaluate_algorithm_parallel(datas[j].urm_test, tested_users, rec)
        #         #     partial_score = partial_score + map
        #         #     print(f"\t\t{j} map : {map}")
        #         #     f.write(f"\t\t{j} map : {map}\n")
        #         # average_score = partial_score / anti_overfitting
        #         print(f"average_map : {average_score}")
        #         f.write(f"average_map : {average_score}\n")
        #         f.flush()
        #         score[int(i)] = (average_score, recommenders[i][0])
        #
        #     # sort the recommenders by their average scores
        #     sorted_scores = sorted(score, key=operator.itemgetter(0), reverse=True)
        #     # take the best ones
        #     best_recs = [t[1] for t in sorted_scores][:int(population_size / 2)]
        #
        #     if epoch != (epochs-1):
        #         for i in range(int(population_size / 2)):
        #             ws = mutate_weights(best_recs[i].weights, epoch=epoch+1)
        #             new_ws = mutate_weights(ws, epoch=epoch+1)
        #             for j in range(anti_overfitting):
        #                 recommenders[i * 2][j] = Hybrid1XXAlphaRecommender(datas[j], base_recommenders[j])
        #                 recommenders[i * 2][j].weights = copy.deepcopy(ws)
        #                 new_rec = Hybrid1XXAlphaRecommender(datas[j], base_recommenders[j])
        #                 new_rec.weights = copy.deepcopy(new_ws)
        #                 recommenders[i * 2 + 1][j] = new_rec
        #
        # for i in range(int(population_size)):
        #     f.write(f"{recommenders[i][0].weights[0]}\n")
        #     f.write(f"{recommenders[i][0].weights[1]}\n")
        #     f.flush()