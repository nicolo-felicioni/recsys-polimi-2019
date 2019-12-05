#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Base.NonPersonalizedRecommender import TopPop, Random
from DataObject import DataObject
from DataReader import DataReader
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender

import traceback

import os, multiprocessing
from functools import partial



# from data.Movielens_10M.Movielens10MReader import Movielens10MReader
from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative


def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """



    #dataReader = Movielens10MReader()
    data_reader = DataReader()


    # URM_train = dataReader.get_URM_train()
    # URM_validation = dataReader.get_URM_validation()
    # URM_test = dataReader.get_URM_test()
    data = DataObject(data_reader, k=1, random_seed=13)

    output_folder_path = "result_experiments_parameter_search/"


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)







    collaborative_algorithm_list = [
        #Random,
        #TopPop,
        P3alphaRecommender,
        RP3betaRecommender,
        #ItemKNNCFRecommender,
        #UserKNNCFRecommender,
        MatrixFactorization_BPR_Cython,
        #MatrixFactorization_FunkSVD_Cython,
        #PureSVDRecommender,
        #SLIM_BPR_Cython,
        #SLIMElasticNetRecommender
    ]




    from Base.Evaluation.Evaluator import EvaluatorHoldout

    evaluator_validation = EvaluatorHoldout(data.urm_test, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(data.urm_train, cutoff_list=[10])


    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train = data.urm_train,
                                                       metric_to_optimize = "MAP",
                                                       n_cases = 10,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = output_folder_path,
                                                       #similarity_type_list = ["cosine"],
                                                       parallelizeKNN = False)





    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

    #
    #
    # for recommender_class in collaborative_algorithm_list:
    #
    #     try:
    #
    #         runParameterSearch_Collaborative_partial(recommender_class)
    #
    #     except Exception as e:
    #
    #         print("On recommender {} Exception {}".format(recommender_class, str(e)))
    #         traceback.print_exc()
    #







if __name__ == '__main__':


    read_data_split_and_search()
