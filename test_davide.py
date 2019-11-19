
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

    dataset_object = RecSys2019Reader()

    dataSplitter = DataSplitter_leave_k_out(dataset_object)

    dataSplitter.load_data()
    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

    target_users = load_target()

    recommender_list = [
        SLIM_BPR_Cython
    ]


    from Base.Evaluation.Evaluator import EvaluatorHoldout

    evaluator = EvaluatorHoldout(URM_test, [10], exclude_seen=True)


    output_root_path = "./result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)


    logFile = open(output_root_path + "result_all_algorithms.txt", "a")
    f = open("submission.csv", "w+")

    for recommender_class in recommender_list:

        try:

            print("Algorithm: {}".format(recommender_class))


            recommender = recommender_class(URM_train)
            recommender.fit()

            for user_id in target_users:
                recommended_items = recommender.recommend(user_id, cutoff=10)
                well_formatted = " ".join([str(x) for x in recommended_items])
                f.write(f"{user_id}, {well_formatted}\n")


        #     results_run, results_run_string = evaluator.evaluateRecommender(recommender)
        #     print("Algorithm: {}, results: \n{}".format(recommender.__class__, results_run_string))
        #     logFile.write("Algorithm: {}, results: \n{}\n".format(recommender.__class__, results_run_string))
        #     logFile.flush()
        #
        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()

