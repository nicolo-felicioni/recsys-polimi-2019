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
    random_seed = 0

    data_reader = DataReader()
    data = DataObject(data_reader, 1, random_seed=random_seed)
    data.print()

    for topK in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
        for shrink in [10, 100, 1000, 10000]:
            for similarity in ["tanimoto", "cosine", "euclidean", "jaccard"]:
                for feature_weighting in ["none", "BM25", "TF-IDF"]:
                    recommender = UserKNNCFRecommender(data.urm_train)
                    recommender.fit(topK=topK,
                                    shrink=shrink,
                                    similarity=similarity,
                                    feature_weighting=feature_weighting)
                    LogToFileEvaluator.evaluate(data,
                                                random_seed,
                                                recommender,
                                                "User CF",
                                                f"topK={topK} - shrink={shrink} - similarity={similarity} - feature_weighting={feature_weighting}",
                                                filename="userCF.csv")

    # for topK in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
    #     for shrink in [10, 100, 1000, 10000]:
    #         for similarity in ["tanimoto", "cosine", "euclidean", "jaccard"]:
    #             for feature_weighting in ["none", "BM25", "TF-IDF"]:
    #                 recommender = ItemKNNCFRecommender(data.urm_train)
    #                 recommender.fit(topK=topK,
    #                                 shrink=shrink,
    #                                 similarity=similarity,
    #                                 feature_weighting=feature_weighting)
    #                 LogToFileEvaluator.evaluate(data,
    #                                             random_seed,
    #                                             recommender,
    #                                             "Item CF",
    #                                             f"topK={topK} - shrink={shrink} - similarity={similarity} - feature_weighting={feature_weighting}",
    #                                             filename="itemCF.csv")

    # for topK in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
    #     for shrink in [1, 2, 5]:
    #         for similarity in ["tanimoto", "cosine", "euclidean", "jaccard"]:
    #             for feature_weighting in ["none", "BM25", "TF-IDF"]:
    #                 recommender = UserKNNCBFRecommender(data.ucm_all, data.urm_train)
    #                 recommender.fit(topK=topK,
    #                                 shrink=shrink,
    #                                 similarity=similarity,
    #                                 feature_weighting=feature_weighting)
    #                 LogToFileEvaluator.evaluate(data,
    #                                             random_seed,
    #                                             recommender,
    #                                             "User CBF Collaborative",
    #                                             f"topK={topK} - shrink={shrink} - similarity={similarity} - feature_weighting={feature_weighting}",
    #                                             filename="userCBF.csv")

    # for topK in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
    #     for shrink in [1, 2, 5]:
    #         for similarity in ["tanimoto", "cosine", "euclidean", "jaccard"]:
    #             for feature_weighting in ["none", "BM25", "TF-IDF"]:
    #                 recommender = ItemKNNCBFRecommender(data.urm_train, data.icm_all_augmented)
    #                 recommender.fit(topK=topK,
    #                                 shrink=shrink,
    #                                 similarity=similarity,
    #                                 feature_weighting=feature_weighting)
    #                 LogToFileEvaluator.evaluate(data,
    #                                             random_seed,
    #                                             recommender,
    #                                             "Item CBF",
    #                                             f"topK={topK} - shrink={shrink} - similarity={similarity} - feature_weighting={feature_weighting}",
    #                                             filename="itemCBF_augmented_all.csv")

    # recommender = SLIM_BPR_Cython(data.urm_train)
    # for lambda_i in [1, 0.1, 0.01, 0.001]:
    #     for lambda_j in [1, 0.1, 0.01, 0.001]:
    #         for learning_rate in [0.1, 0.001]:
    #             for epochs in [50, 100, 150, 250, 400, 600, 1000, 1500, 2500]:
    #                 for topK in [10, 50, 200, 1000, 5000]:
    #                     for sgd_mode in ["adagrad", "sgd", "adam", "rmsprop"]:
    #                         recommender.fit(topK=topK,
    #                                         epochs=epochs,
    #                                         lambda_i=lambda_i,
    #                                         lambda_j=lambda_j,
    #                                         learning_rate=learning_rate,
    #                                         sgd_mode=sgd_mode)
    #                         LogToFileEvaluator.evaluate(data,
    #                                                     random_seed,
    #                                                     recommender,
    #                                                     "SLIM_BPR",
    #                                                     f"sgd_mode={sgd_mode} - topK={topK} - epochs={epochs} - learning_rate={learning_rate} - lambda_i={lambda_i} - lambda_j={lambda_j}",
    #                                                     filename="slim_bpr.csv")

    # for topK in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
    #     for alpha in range(0, 100, 5):
    #         alpha = alpha / 100
    #         for beta in range(0, 100, 5):
    #             beta = beta / 100
    #             for implicit in [True]:
    #                 recommender = RP3betaRecommender(data.urm_train)
    #                 recommender.fit(topK=topK,
    #                                 alpha=alpha,
    #                                 beta=beta,
    #                                 implicit=implicit)
    #                 LogToFileEvaluator.evaluate(data,
    #                                             random_seed,
    #                                             recommender,
    #                                             "RP3",
    #                                             f"topK={topK} - alpha={alpha} - beta={beta} - implicit={implicit}",
    #                                             filename="rp3.csv")

    # recommender = PureSVDRecommender(data.urm_train)
    # for num_factors in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
    #     recommender.fit(num_factors=num_factors)
    #     LogToFileEvaluator.evaluate(data,
    #                                 random_seed,
    #                                 recommender,
    #                                 "PureSVD",
    #                                 f"num_factors={num_factors}",
    #                                 filename="pure_svd.csv")