import operator
import copy
import math
import time

from joblib import Parallel, delayed

from Base.Evaluation import MyEvaluator, LogToFileEvaluator
from Base.Evaluation.Evaluator import EvaluatorHoldout, Evaluator
from Custom.AUserToMultipleUserRecommender import AUserToMultipleUserRecommender
from DataObject import DataObject
from DataReader import DataReader
from FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
from Hybrid.Hybrid000AlphaRecommender import Hybrid000AlphaRecommender
from Hybrid.Hybrid001AlphaRecommender import Hybrid001AlphaRecommender
from Hybrid.Hybrid003AlphaRecommender import Hybrid003AlphaRecommender
from Hybrid.Hybrid004AlphaRecommender import Hybrid004AlphaRecommender
from Hybrid.Hybrid100AlphaRecommender import Hybrid100AlphaRecommender
from Hybrid.Hybrid101AlphaRecommender import Hybrid101AlphaRecommender
from Hybrid.Hybrid102AlphaRecommender import Hybrid102AlphaRecommender
from Hybrid.Hybrid105AlphaRecommender import Hybrid105AlphaRecommender
from Hybrid.Hybrid108AlphaRecommender import Hybrid108AlphaRecommender
from Hybrid.Hybrid109AlphaRecommender import Hybrid109AlphaRecommender
from Hybrid.Hybrid1CXAlphaRecommender import Hybrid1CXAlphaRecommender
from Hybrid.Hybrid1CYAlphaRecommender import Hybrid1CYAlphaRecommender
from Hybrid.Hybrid1XXAlphaRecommender import Hybrid1XXAlphaRecommender
from Hybrid.Hybrid200AlphaRecommender import Hybrid200AlphaRecommender
from KNN.ItemKNNCBFOnlyColdRecommender import ItemKNNCBFOnlyColdRecommender
from KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from MatrixFactorization.IALSRecommender import IALSRecommender
from MatrixFactorization.MatrixFactorization_BPR_Theano import MatrixFactorization_BPR_Theano
from MatrixFactorization.NMFRecommender import NMFRecommender
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


def mutate_weights(weights, mutation_factor=0.5, epoch=1, min_mutation=0.005, big_mutation_probability=0.0001,
                   big_mutation_factor=10, k=2):
    big_mutation_probability = big_mutation_probability / k
    big_mutation_factor = big_mutation_factor * k
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
            if col > 0:
                weights[row][col] = max(weights[row][col], weights[row][col - 1])
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
    data = DataObject(data_reader, k=0, random_seed=999)
    data.urm_train = data.urm
    recommender = Hybrid004AlphaRecommender(data)
    recommender.fit()
    f = open("submission.csv", "w+")
    f.write("user_id,item_list\n")
    for user_id in data.ids_target_users:
        recommended_items = recommender.recommend(user_id, cutoff=10)
        print(user_id)
        print(recommended_items)
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

        random_seed = 0

        data_reader = DataReader()
        data = DataObject(data_reader, 1, random_seed=random_seed)
        data.print()

        recommender = AUserToMultipleUserRecommender(data)
        r1 = RP3betaRecommender(recommender.new_urm_train)
        r1.fit(topK=20, alpha=0.16, beta=0.24)
        recommender.fit(r1)
        for n, users, description in data.urm_train_users_by_type:
            eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
            print(f"ALL 10,\t{n}, \t{description},\t {eval}")
        users = data.ids_target_users
        eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
        print(f"ALL 10,\t,\t {eval}")
        users = data.ids_user
        eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
        print(f"ALL 10,\t,\t {eval}")

        # r1 = RP3betaRecommender(data.urm_train)
        # r1.fit(topK=20, alpha=0.16, beta=0.24)
        # s1 = r1.W_sparse
        # #s1 = s1 / s1.data.max()
        # r2 = ItemKNNCBFRecommender(data.urm_train, data.icm_all_augmented)
        # r2.fit(shrink=30, topK=20, similarity="jaccard", feature_weighting="BM25")
        # s2 = r2.W_sparse
        # # s2.data = s2.data / s2.data.max()
        #
        # recommender = ItemKNNSimilarityHybridRecommender(data.urm_train, s1, s2)
        # recommender.fit(alpha=0.5)
        #
        # for n, users, description in data.urm_train_users_by_type:
        #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
        #     print(f"ALL 10,\t{n}, \t{description},\t {eval}")
        # users = data.ids_target_users
        # eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
        # print(f"ALL 10,\t,\t {eval}")
        # users = data.ids_user
        # eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
        # print(f"ALL 10,\t,\t {eval}")
        #
        # item_list_row = []
        # item_list_col = []
        # data_list = []
        #
        # all_cold_item = np.append(data.ids_cold_item, data.ids_cold_train_items)
        #
        # for cold_item in all_cold_item:
        #     cols = s2[cold_item].indices
        #     ds = s2[cold_item].data
        #     for col, d in zip(cols, ds):
        #         item_list_row.append(cold_item)
        #         item_list_col.append(col)
        #         data_list.append(d)
        # new_s2 = sps.csr_matrix((data_list, (item_list_col, item_list_row)), shape=s1.shape)
        #

        # recommender = UserKNNCFRecommender(data.urm_train)
        # recommender.fit(topK=2000, shrink=100, similarity="cosine", feature_weighting="BM25")
        # data.remove_close_to_cold_item_interactions(1)

        # recommender = Hybrid109AlphaRecommender(data)
        # recommender.fit()
        # for n, users, description in data.urm_train_users_by_type:
        #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
        #     print(f"ALL 10,\t{n}, \t{description},\t {eval}")
        # users = data.ids_target_users
        # eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
        # print(f"ALL 10,\t,\t {eval}")
        # users = data.ids_user
        # eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
        # print(f"ALL 10,\t,\t {eval}")

        # recommender = Hybrid003AlphaRecommender(data)
        # recommender.fit()
        # for n, users, description in data.urm_train_users_by_type:
        #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
        #     print(f"ALL 10,\t,\t {eval}")
        # users = data.ids_target_users
        # eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
        # print(f"ALL 10,\t,\t {eval}")

        # recommender = UserKNNCBFRecommender(data.ucm_all, data.urm_train)
        # recommender.fit(topK=1000, shrink=1)
        # n, users, description = data.urm_train_users_by_type[0]
        # eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
        # print(f"ALL 10,\t,\t {eval}")

        # icm1 = data.icm_asset_augmented
        # icm2 = data.icm_price_augmented
        # icm3 = data.icm_classZ
        #
        # icm_all = sps.hstack((icm1, icm2, icm3))
        #
        # recommender = ItemKNNCBFOnlyColdRecommender(data)
        # recommender.fit(topK=10, shrink=20)
        # # recommender = P3alphaRecommender(data.urm_train)
        # # recommender.fit()
        # for n, users, description in data.urm_train_users_by_type:
        #
        #     cumulative_MAP = 0.0
        #     userList_unique = users
        #     URM_test = data.urm_test
        #     num_eval = 0.00001
        #     size = 0
        #     for user_id in userList_unique:
        #         relevant_items = URM_test[user_id].indices
        #         if len(relevant_items):
        #             recommended_items = recommender.recommend(user_id, cutoff=4)
        #             recommended_items = [x for x in recommended_items if x in data.ids_cold_train_items or x in data.ids_cold_item]
        #             size += len(recommended_items)
        #             if len(recommended_items):
        #                 _m = MyEvaluator.MAP(recommended_items, relevant_items)
        #                 cumulative_MAP += _m
        #                 num_eval += 1
        #     cumulative_MAP /= num_eval
        #     print(f"{size/users.shape[0]}")
        #
        #     print(f"ALL 10,\t {description},\t {cumulative_MAP}")
        # users = data.ids_warm_user
        # eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
        # print(f"ALL 10,\t,\t {eval}")

        # rec1 = RP3betaRecommender(data.urm_train)
        # rec1.fit(topK=150, alpha=0.55, beta=0.05, normalize_similarity=True, implicit=True)
        # rec2 = ItemKNNCFRecommender(data.urm_train)
        # rec2.fit(topK=20000, shrink=5000, similarity="tanimoto")

        # rec1 = UserKNNCFRecommender(data.urm_train)
        # rec1.fit(topK=500, shrink=500, feature_weighting="TF-IDF")
        # rec2 = ItemKNNCFRecommender(data.urm_train)
        # rec2.fit(topK=5, shrink=29, similarity="jaccard")
        # rec3 = RP3betaRecommender(data.urm_train)
        # rec3.fit(topK=100, alpha=0.55, beta=0.01, implicit=True, normalize_similarity=True)
        # for n, users, description in data.urm_train_users_by_type:
        #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec1, at=10, remove_top=0)
        #     print(f"ALL 10,\t {description},\t {eval}")
        #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec2, at=10, remove_top=0)
        #     print(f"ALL 10,\t {description},\t {eval}")
        #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec3, at=10, remove_top=0)
        #     print(f"ALL 10,\t {description},\t {eval}")
        #     average_length = 0
        #     for user in users:
        #         items1 = rec1.recommend(user, cutoff=10)
        #         items2 = rec2.recommend(user, cutoff=10)
        #         items3 = rec3.recommend(user, cutoff=10)
        #         if(len(items1) > 0 and len(items2) > 0 and len(items3) > 0):
        #             all_items = [*items1, *items2, *items3]
        #             length = np.unique(all_items).shape[0]
        #             average_length += length
        #     print(f"average_length : {average_length / n}")
        # users = data.ids_warm_user
        # eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec1, at=10, remove_top=0)
        # print(f"ALL 10,\t,\t {eval}")

        # rec1 = UserKNNCFRecommender(data.urm_train)
        # rec1.fit(topK=500, shrink=500, feature_weighting="TF-IDF")
        # rec2 = ItemKNNCFRecommender(data.urm_train)
        # rec2.fit(topK=5, shrink=29, similarity="jaccard")
        # for n, users, description in data.urm_train_users_by_type:
        #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec1, at=10, remove_top=0)
        #     print(f"ALL 10,\t {description},\t {eval}")
        #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec2, at=10, remove_top=0)
        #     print(f"ALL 10,\t {description},\t {eval}")
        #     average_length = 0
        #     for user in users:
        #         items1 = rec1.recommend(user, cutoff=10)
        #         items2 = rec2.recommend(user, cutoff=10)
        #         # items3 = rec3.recommend(user, cutoff=10)
        #         if(len(items1) > 0 and len(items2) > 0):
        #             all_items = [*items1, *items2]
        #             length = np.unique(all_items).shape[0]
        #             average_length += length
        #     print(f"average_length : {average_length / n}")

        # recommender = ItemKNNCFRecommender(data.urm_train)
        # recommender.fit(topK=12, shrink=15, feature_weighting="none", similarity="jaccard")
        # n, users, description = data.urm_train_users_by_type[8]
        # eval, map = MyEvaluator.evaluate_algorithm_parallel(data.urm_test, users, recommender, at=10, remove_top=0, parallelism=8)
        # print(f"ALL 10,\t {description},\t {eval}")

        # ws = []
        # m_cutoff = 30
        # anti_overfitting_generation = 20
        # for i in range(anti_overfitting_generation):
        #     data_reader = DataReader()
        #     data = DataObject(data_reader, 1, random_seed=(20+i))
        #     rec1 = SLIM_BPR_Cython(data.urm_train)
        #     rec2 = ItemKNNCFRecommender(data.urm_train)
        #     rec3 = RP3betaRecommender(data.urm_train)
        #     rec1.fit(sgd_mode="adagrad", topK=30, epochs=150, learning_rate=1e-05, lambda_i=1, lambda_j=0.001)
        #     rec2.fit(topK=10, shrink=30, similarity="tanimoto")
        #     rec3.fit(topK=20, alpha=0.16, beta=0.24, implicit=True, normalize_similarity=True)
        #     # rec1 = ItemKNNCFRecommender(data.urm_train)
        #     # rec1.fit(topK=30, shrink=30, similarity="jaccard")
        #     # rec1 = ItemKNNCFRecommender(data.urm_train)
        #     # rec1.fit(topK=20000, shrink=20000,feature_weighting="TF-IDF")
        #     # rec2 = ItemKNNCBFOnlyColdRecommender(data)
        #     # rec2.fit(topK=20)
        #     base_recommenders = [rec1, rec2, rec3]
        #     # t_users = data.ids_user
        #     n, t_users, description = data.urm_train_users_by_type[5]
        #     tested_users = t_users
        #     rec = Hybrid1CYAlphaRecommender(data, base_recommenders, tested_users, max_cutoff=m_cutoff)
        #     ws.append(rec.weights)
        # mean_ws = np.zeros(shape=ws[0].shape)
        # for i in range(anti_overfitting_generation):
        #     mean_ws += ws[i]
        # mean_ws = mean_ws / anti_overfitting_generation
        #
        # rec = Hybrid1XXAlphaRecommender(data, base_recommenders, max_cutoff=m_cutoff)
        # rec.fit(weights=mean_ws)
        # # # for n, users, description in data.urm_train_users_by_type:
        # # #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec, at=10, remove_top=0)
        # # #     print(f"ALL 10,\t {description},\t {eval}")
        # # n, t_users, description = data.urm_train_users_by_type[0]
        # # # t_users = data.ids_target_users
        # # eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, t_users, rec, at=10, remove_top=0)
        # # print(f"ALL 10,\t {description},\t {eval}")
        #
        # print(mean_ws[0])
        # print(mean_ws[1])
        # f = open("weights", "a+")
        # f.write("General, \n \n")
        # f.write(f"0 -> Slim, topK=30, epochs=150, learning_rate=1e-05, lambda_i=1, lambda_j=0.001\n")
        # f.write(f"1-> UserCBF, topK=10, shrink=30, similarity=tanimoto\n")
        # f.write(f"2 -> RP3, topK=20, alpha=0.16, beta=0.24, implicit=True, normalize_similarity=True\n")
        # f.write(f"{mean_ws[0]}\n")
        # f.write(f"{mean_ws[1]}\n")
        # f.flush()
        # for n, users, description in data.urm_train_users_by_type:
        #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec, at=10, remove_top=0)
        #     print(f"ALL 10,\t {description},\t {eval}")
        # users = data.ids_warm_user
        # eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec, at=10, remove_top=0)
        # print(f"ALL 10,\t,\t {eval}")

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
        # recommender = PureSVDRecommender(data.urm_train)
        # recommender.fit(num_factors=5)

        # LogToFileEvaluator.evaluate(data, random_seed, recommender, "PureSVD", "num_factors=5")
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

        # f = open("submission_old.csv", "w+")
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
        # f = open("submission_old.csv", "w+")
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
        # epochs = 20
        # score = [None] * population_size
        # anti_overfitting = 2
        #
        # # recommenders[population_id][anti_overfitting_generation]
        # recommenders = np.full((population_size, anti_overfitting), None)
        # # base_recommenders[anti_overfitting_generation]
        # base_recommenders = []
        # tested_recommenders = []
        # datas = [None] * anti_overfitting
        # tested_users = []
        # rec_model = []
        #
        # for j in range(anti_overfitting):
        #     data_reader = DataReader()
        #     datas[j] = DataObject(data_reader, 1, random_seed=(50 + j*10))
        #     rec1 = ItemKNNCFRecommender(datas[j].urm_train)
        #     rec1.fit(shrink=15, topK=12, feature_weighting="none")
        #     rec2 = P3alphaRecommender(datas[j].urm_train)
        #     rec2.fit(topK=170, implicit=True, alpha=0.5)
        #     # rec3 = SLIM_BPR_Cython(datas[j].urm_train)
        #     # rec3.fit(epochs=150, topK=1000, lambda_i=0.15)
        #     rec3 = RP3betaRecommender(datas[j].urm_train)
        #     rec3.fit(topK=60, alpha=0.5, beta=0.1, implicit=True)
        #     rec4 = ItemKNNCFRecommender(datas[j].urm_train)
        #     rec4.fit(shrink=30, topK=30, feature_weighting="none", similarity="tanimoto")
        #     base_recommenders.append([rec1, rec2, rec3, rec4])
        #     n, t_users, description = datas[j].urm_train_users_by_type[8]
        #     tested_users.append(t_users)
        #     rec_model.append(Hybrid1CXAlphaRecommender(datas[j], base_recommenders[j], tested_users[j]))
        #
        # k_factor = len(base_recommenders[0])
        #
        # for i in range(population_size):
        #     for j in range(anti_overfitting):
        #         rec = rec_model[j].clone()
        #         recommenders[i][j] = rec
        #
        # for i in range(population_size):
        #     ws = mutate_weights(recommenders[i][0].weights, k=k_factor)
        #     for j in range(anti_overfitting):
        #         recommenders[i][j].weights = ws
        #
        # start_time = time.time()
        #
        #
        # for epoch in range(epochs):
        #     print(f"epoch {epoch}")
        #     f.write(f"epoch {epoch}\n")
        #     print("Epoch {} of {} complete in {:.2f} minutes\n".format(epoch, epochs,
        #                                                              float(time.time() - start_time) / 60))
        #     f.write("Epoch {} of {} complete in {:.2f} minutes\n".format(epoch, epochs,
        #                                                              float(time.time() - start_time) / 60))
        #     start_time = time.time()
        #
        #     for i in range(population_size):
        #         # partial_score = 0
        #         def parallel_run(j):
        #             _rec = recommenders[i][j]
        #             _tested_users = tested_users[j]
        #             _result_string, _map = MyEvaluator.evaluate_algorithm(datas[j].urm_test, _tested_users, _rec)
        #             print(f"\t\t{j} map : {_map}")
        #             f.write(f"\t\t{j} map : {_map}\n")
        #             return _map
        #         partial_scores = Parallel(n_jobs=anti_overfitting)(delayed(parallel_run)(generation) for generation in range(anti_overfitting))
        #         average_score = 0
        #         for s in partial_scores:
        #             average_score += s
        #         average_score = average_score / anti_overfitting
        #         print(f"\taverage_map : {average_score}")
        #         f.write(f"\taverage_map : {average_score}\n")
        #         standard_deviation = 0
        #         for s in partial_scores:
        #             standard_deviation += (s-average_score)*(s-average_score)
        #         standard_deviation = math.sqrt(standard_deviation / (anti_overfitting - 1))
        #         print(f"\tstandard_deviation : {standard_deviation}")
        #         f.write(f"\tstandard_deviation : {standard_deviation}\n")
        #         average_score = average_score - (standard_deviation/3)*2
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
        #         print(f"average_score : {average_score}")
        #         f.write(f"average_score : {average_score}\n")
        #         f.flush()
        #         score[int(i)] = (average_score, recommenders[i])
        #
        #     # sort the recommenders by their average scores
        #     sorted_scores = sorted(score, key=operator.itemgetter(0), reverse=True)
        #     # take the best ones
        #     best_recs = [t[1] for t in sorted_scores][:int(population_size / 2)]
        #
        #     if epoch != (epochs-1):
        #         for i in range(int(population_size / 2)):
        #             ws = mutate_weights(best_recs[i][j].weights, epoch=epoch+1, k=k_factor)
        #             new_ws = mutate_weights(ws, epoch=epoch+1, k=k_factor)
        #             for j in range(anti_overfitting):
        #                 recommenders[i * 2][j] = best_recs[i][j].clone()
        #                 recommenders[i * 2][j].weights = copy.deepcopy(ws)
        #                 new_rec = best_recs[i][j].clone()
        #                 new_rec.weights = copy.deepcopy(new_ws)
        #                 recommenders[i * 2 + 1][j] = new_rec
        #
        # for i in range(int(population_size)):
        #     for w in recommenders[i][0].weights:
        #         f.write(f"w {i}\n -> {w} \n")
        #         f.flush()
