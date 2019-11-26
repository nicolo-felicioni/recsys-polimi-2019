import time

import numpy as np
from Base.Evaluation import MyEvaluator
from DataObject import DataObject
from DataReader import DataReader
from Hybrid.Hybrid001AlphaRecommender import Hybrid001AlphaRecommender
from Hybrid.Hybrid002AlphaRecommender import Hybrid002AlphaRecommender
from Hybrid.Hybrid003AlphaRecommender import Hybrid003AlphaRecommender
from KNN.UserKNNAgeRecommender import UserKNNAgeRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
import pandas as pd
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sps
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
#from MatrixFactorization.WRMFRecommender import WRMFRecommender

from Base.NonPersonalizedRecommender import TopPop, Random, GlobalEffects

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender

from Data_manager.RecSys2019.RecSys2019Reader import RecSys2019Reader
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out

import traceback, os


def make_submission():

    print("making the submission")
    data_reader_loc = DataReader()
    data_loc = DataObject(data_reader_loc, k=0)
    recommender = Hybrid003AlphaRecommender(data_loc)
    recommender.fit()
    f = open("submissionHybrid003_k_12_shrink_15.csv", "w+")
    f.write("user_id,item_list\n")
    for user_id in data_loc.ids_target_users:
        recommended_items = recommender.recommend(user_id, cutoff=10)
        well_formatted = " ".join([str(x) for x in recommended_items])
        f.write(f"{user_id}, {well_formatted}\n")


def slim_tuning(f, epochs=300,
            train_with_sparse_weights = None,
            symmetric = True,
            verbose = False,
            random_seed = None,
            batch_size = 1000, lambda_i = 0.0, lambda_j = 0.0, learning_rate = 1e-4, topK = 200,
            sgd_mode='adagrad', gamma=0.995, beta_1=0.9, beta_2=0.999):
    seed = 13
    map_dict = {}
    description_list = []
    flag = 0
    k_fold = 2

    for i in range(0, k_fold):

        data_reader = DataReader()
        data = DataObject(data_reader, k=1, random_seed=seed)

        rec = SLIM_BPR_Cython(data.urm_train)

        rec.fit(epochs=epochs,
                train_with_sparse_weights=train_with_sparse_weights,
                symmetric=symmetric,
                verbose=verbose,
                random_seed=random_seed,
                batch_size=batch_size, lambda_i=lambda_i, lambda_j=lambda_j, learning_rate=learning_rate, topK=topK,
                sgd_mode=sgd_mode, gamma=gamma, beta_1=beta_1, beta_2=beta_2)

        # initializing the description list
        if (len(description_list) == 0):
            for _, _, description in data.urm_train_users_by_type:
                description_list.append(description)

        # initializing the dictionary
        if (flag == 0):
            flag = 1
            for d in description_list:
                map_dict[d] = 0


        for n, users, description in data.urm_train_users_by_type:
            eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec, at=10, remove_top=0)
            print(f"\t {description},\t {eval}")
            map_dict[description] += map

    print("epochs={}, train_with_sparse_weights = {},symmetric = {}, batch_size = {},".format(epochs,
                                                                                              train_with_sparse_weights,
                                                                                              symmetric, batch_size))
    print("lambda_i = {}, lambda_j = {}, learning_rate = {}, topK = {},".format(lambda_i,lambda_j,learning_rate,topK))
    print("sgd_mode={}, sgd_mode={}, beta_1={}, beta_2={}".format(sgd_mode,sgd_mode,beta_1,beta_2))
    for d in description_list:
        map_dict[d] /= k_fold
        print(d + "\t\t" + "average map: " + str(map_dict[d]))


    f.write("\nepochs={}, train_with_sparse_weights = {},symmetric = {}, batch_size = {},".format(epochs,
                                                                                              train_with_sparse_weights,
                                                                                              symmetric, batch_size))
    f.write("\nlambda_i = {}, lambda_j = {}, learning_rate = {}, topK = {},".format(lambda_i,lambda_j,learning_rate,topK))
    f.write("\nsgd_mode={}, sgd_mode={}, beta_1={}, beta_2={}".format(sgd_mode,sgd_mode,beta_1,beta_2))
    for d in description_list:
        f.write("\n" + d + "\t\t" + "average map: " + str(map_dict[d]))
    f.write("\n\n")
    f.flush()






if __name__ == '__main__':

    # df_user_age = pd.read_csv("Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_UCM_age.csv")


    want_submission = False
    if want_submission:
        make_submission()
    else:
        # epoch_list = [50, 100, 150, 200]
        # for e in epoch_list:
        #     slim_tuning(epochs=e) # we fix epoch = 150
        want_slim_tuning = False
        if want_slim_tuning:

            start_time = time.time()

            # lambda_j_list = [0.01, 0.05, 0.1, 0.2]
            # lambda_i_list = [0]
            f = open("slim_high_reg_more_epochs.txt", "w+")
            f.write("final review\n\n")
            slim_tuning(f, epochs=200, lambda_i=0.7, lambda_j=1, sgd_mode='adagrad')  # we fix epoch = 150

            print("Completed in {:.2f} minutes".format(float(time.time() - start_time)/60))

            f.flush()
            f.close()

            # slim_tuning(epochs=300,
            #     train_with_sparse_weights = None,
            #     symmetric = True,
            #     verbose = False,
            #     random_seed = None,
            #     batch_size = 1000, lambda_i = 0.0, lambda_j = 0.0, learning_rate = 1e-4, topK = 200,
            #     sgd_mode='adagrad', gamma=0.995, beta_1=0.9, beta_2=0.999)
        else:
            data_reader = DataReader()
            data = DataObject(data_reader, k=1)
            rec = SLIM_BPR_Cython(data.urm_train)
            rec.fit(epochs=200, lambda_i=0.7, lambda_j=1, sgd_mode='adagrad')
            print(MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_user, rec))

        # rec1 = Hybrid003AlphaRecommender(data)
        # rec1.fit()
        # print(MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_user, rec1))
        #rec2 = Hybrid002AlphaRecommender(data.urm_train, data.ucm_region, data.ids_cold_train_users, data.ids_warm_train_users)
        #rec2.fit()
        #MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_cold_train_users, rec2)

    '''
    data_reader = DataReader()
    data = DataObject(data_reader, k=0)
    print(type(data.ids_target_users))
    recommender = Hybrid001AlphaRecommender(data.urm_train, data.ucm_region, data.ids_cold_train_users,
                                     data.ids_warm_train_users)
    recommender.fit()
    f = open("submissionHybrid001.csv", "w+")
    f.write("user_id,item_list\n")

    for user_id in data.ids_target_users:
        recommended_items = recommender.recommend(user_id, cutoff=10)
        well_formatted = " ".join([str(x) for x in recommended_items])
        f.write(f"{user_id}, {well_formatted}\n")

    '''


