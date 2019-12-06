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
from KNN.NewUserKNNAgeRecommender import NewUserKNNAgeRecommender
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

    # TODO: Edit here
    max_cutoff = 30
    # TODO: Edit here
    anti_overfitting_generation = 1

    ws = []
    base_recommenders = []
    data = None
    description_list = []

    for i in range(anti_overfitting_generation):
        data_reader = DataReader()
        data = DataObject(data_reader, 1, random_seed=(20 + i))

        # TODO: Edit here
        # Change the recommenders
        rec1 = SLIM_BPR_Cython(data.urm_train)
        rec1.fit(sgd_mode="adagrad", topK=30, epochs=150, learning_rate=1e-05, lambda_i=1, lambda_j=0.001)
        description_list.append(f"SLIM_BPR sgd_mode=adagrad, topK=30, epochs=150, learning_rate=1e-05, lambda_i=1, lambda_j=0.001")
        rec2 = ItemKNNCFRecommender(data.urm_train)
        rec2.fit(topK=10, shrink=30, similarity="tanimoto")
        description_list.append(f"Item CF topK=10, shrink=30, similarity=tanimoto")
        rec3 = RP3betaRecommender(data.urm_train)
        rec3.fit(topK=20, alpha=0.16, beta=0.24, implicit=True, normalize_similarity=True)
        description_list.append(f"Item CF topK=20, alpha=0.16, beta=0.24, implicit=True, normalize_similarity=True")

        # TODO: Edit here
        # Target of the tuning is type [ ... ]
        n, t_users, description = data.urm_train_users_by_type[5]

        base_recommenders = [rec1, rec2, rec3]
        tested_users = t_users
        rec = Hybrid1CYAlphaRecommender(data, base_recommenders, tested_users, max_cutoff=max_cutoff)
        ws.append(rec.weights)
    mean_ws = np.zeros(shape=ws[0].shape)
    for i in range(anti_overfitting_generation):
        mean_ws += ws[i]
    mean_ws = mean_ws / anti_overfitting_generation

    f = open("weights", "a+")
    for i in range(0, len(base_recommenders)):
        f.write(f"{description_list[i]}\n")
        f.write(f"{mean_ws[i].tolist()}\n")
    f.flush()
