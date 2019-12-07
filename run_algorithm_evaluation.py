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
    for seed in [0, 1, 2, 3, 4]:
        data_reader = DataReader()
        data = DataObject(data_reader, 1, random_seed=seed)
        recommender = RP3betaRecommender(data.urm_train)
        recommender.fit(topK=10, alpha=0.27, beta=0.11)
        LogToFileEvaluator.evaluate(data,
                                    seed,
                                    recommender,
                                    "RP3",
                                    "",
                                    filename="algo_eval.csv")

    # for seed in [0, 1, 2, 3, 4]:
    #     data_reader = DataReader()
    #     data = DataObject(data_reader, 1, random_seed=seed)
    #     recommender = ItemKNNCFRecommender(data.urm_train)
    #     recommender.fit(topK=22, shrink=850, similarity="jaccard", feature_weighting="BM25")
    #     LogToFileEvaluator.evaluate(data,
    #                                 seed,
    #                                 recommender,
    #                                 "ITEM",
    #                                 "",
    #                                 filename="algo_eval.csv")

    # for seed in [0, 1, 2, 3, 4]:
    #     data_reader = DataReader()
    #     data = DataObject(data_reader, 1, random_seed=seed)
    #     recommender = UserKNNCFRecommender(data.urm_train)
    #     recommender.fit(topK=2000, shrink=10, similarity="jaccard", feature_weighting="none")
    #     LogToFileEvaluator.evaluate(data,
    #                                 seed,
    #                                 recommender,
    #                                 "ITEM",
    #                                 "",
    #                                 filename="algo_eval.csv")

    # for seed in [0, 1, 2, 3, 4]:
    #     data_reader = DataReader()
    #     data = DataObject(data_reader, 1, random_seed=seed)
    #     recommender = Hybrid100AlphaRecommender(data)
    #     recommender.fit()
    #     LogToFileEvaluator.evaluate(data,
    #                                 seed,
    #                                 recommender,
    #                                 "H100",
    #                                 "",
    #                                 filename="algo_eval.csv")