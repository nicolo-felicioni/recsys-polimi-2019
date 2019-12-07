import operator
import copy
import math
import time

from joblib import Parallel, delayed

from Base.Evaluation import MyEvaluator, LogToFileEvaluator
from Base.Evaluation.Evaluator import EvaluatorHoldout, Evaluator
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

if __name__ == '__main__':

    # n_dataset = 10
    # parallelism = 3
    # def eval_one_data(random_seed):
    #     data_reader = DataReader()
    #     data = DataObject(data_reader, 1, random_seed=random_seed)
    #     result = []
    #     recommender = UserKNNCBFRecommender(data.ucm_all, data.urm_train)
    #     recommender.fit(topK = args["topK"], shrink=args["shrink"], similarity=args["similarity"], feature_weighting=args["feature_weighting"])
    #     for n, users, description in data.urm_train_users_by_type:
    #         eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
    #         result.append(map)
    #     users = data.ids_target_users
    #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
    #     result.append(map)
    #     return result
    # for topK in range(2000, 7500, 250):
    #     for shrink in [1,2,5]:
    #         for similarity in ["euclidean"]:
    #             for feature_weighting in ["TF-IDF", "none"]:
    #                 args = {"topK": topK, "shrink": shrink, "similarity": similarity,
    #                         "feature_weighting": feature_weighting}
    #                 partial_scores = Parallel(n_jobs=parallelism)(
    #                     delayed(eval_one_data)(generation) for generation in range(15, 15 + n_dataset))
    #                 mean_score = np.zeros(shape=len(partial_scores[0]))
    #                 for s in partial_scores:
    #                     mean_score += s
    #                 mean_score = mean_score / n_dataset
    #                 f = open("deep_user_cbf_euclidean_cold_users.csv", "a+")
    #                 values = " ".join([str(x) + "," for x in mean_score])
    #                 f.write(f"{args}, {values}\n")
    #                 f.flush()

    n_dataset = 10
    def eval_one_data(random_seed):
        data_reader = DataReader()
        data = DataObject(data_reader, 1, random_seed=random_seed)
        result = []
        recommender = ItemKNNCFRecommender(data.urm_train)
        recommender.fit(topK = args["topK"], shrink=args["shrink"], similarity=args["similarity"], feature_weighting=args["feature_weighting"])
        for n, users, description in data.urm_train_users_by_type:
            eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
            result.append(map)
        users = data.ids_target_users
        eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
        result.append(map)
        return result
    for topK in range(5, 150, 5):
        for shrink in range(10, 150, 10):
            for similarity in ["tanimoto"]:
                for feature_weighting in ["BM25"]:
                    args = {"topK": topK, "shrink": shrink, "similarity": similarity,
                            "feature_weighting": feature_weighting}
                    partial_scores = Parallel(n_jobs=n_dataset)(
                        delayed(eval_one_data)(generation) for generation in range(15, 15 + n_dataset))
                    mean_score = np.zeros(shape=len(partial_scores[0]))
                    for s in partial_scores:
                        mean_score += s
                    mean_score = mean_score / n_dataset
                    f = open("deep_item_cf_tanimoto_bm25.csv", "a+")
                    values = " ".join([str(x) + "," for x in mean_score])
                    f.write(f"{args}, {values}\n")
                    f.flush()


    # n_dataset = 10
    # def eval_one_data(random_seed):
    #     data_reader = DataReader()
    #     data = DataObject(data_reader, 1, random_seed=random_seed)
    #     result = []
    #     recommender = ItemKNNCFRecommender(data.urm_train)
    #     recommender.fit(topK = args["topK"], shrink=args["shrink"], similarity=args["similarity"], feature_weighting=args["feature_weighting"])
    #     for n, users, description in data.urm_train_users_by_type:
    #         eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
    #         result.append(map)
    #         users = data.ids_target_users
    #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
    #     result.append(map)
    #     return result
    # for topK in range(5000, 10000, 500):
    #     for shrink in range(5000, 10000, 500):
    #         for similarity in ["cosine"]:
    #             for feature_weighting in ["TF-IDF"]:
    #                 args = {"topK": topK, "shrink": shrink, "similarity": similarity,
    #                         "feature_weighting": feature_weighting}
    #                 partial_scores = Parallel(n_jobs=n_dataset)(
    #                     delayed(eval_one_data)(generation) for generation in range(15, 15 + n_dataset))
    #                 mean_score = np.zeros(shape=len(partial_scores[0]))
    #                 for s in partial_scores:
    #                     mean_score += s
    #                 mean_score = mean_score / n_dataset
    #                 f = open("eval_3.csv", "a+")
    #                 values = " ".join([str(x) + "," for x in mean_score])
    #                 f.write(f"{args}, {values}\n")
    #                 f.flush()

    # n_dataset = 10
    # def eval_one_data(random_seed):
    #     data_reader = DataReader()
    #     data = DataObject(data_reader, 1, random_seed=random_seed)
    #     result = []
    #     recommender = RP3betaRecommender(data.urm_train)
    #     recommender.fit(topK = args["topK"], alpha=args["alpha"], beta=args["beta"])
    #     for n, users, description in data.urm_train_users_by_type:
    #         eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
    #         result.append(map)
    #     users = data.ids_target_users
    #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
    #     result.append(map)
    #     return result
    #
    #
    # for topK in range(5, 40):
    #     for alpha in range(5, 35, 2):
    #         alpha = alpha / 100
    #         for beta in range(5, 35, 2):
    #             beta = beta / 100
    #             args = {"topK": topK, "alpha": alpha, "beta": beta}
    #             partial_scores = Parallel(n_jobs=n_dataset)(
    #                 delayed(eval_one_data)(generation) for generation in range(15, 15 + n_dataset))
    #             mean_score = np.zeros(shape=len(partial_scores[0]))
    #             for s in partial_scores:
    #                 mean_score += s
    #             mean_score = mean_score / n_dataset
    #             f = open("deep_rp3_low_alpha_and_beta.csv", "a+")
    #             values = " ".join([str(x) + "," for x in mean_score])
    #             f.write(f"{args}, {values}\n")
    #             f.flush()