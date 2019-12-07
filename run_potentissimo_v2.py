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


# Mutate an input weights. Has no side effect on the input matrix
def mutate_weights(weights,
                   # How much a weight can mutate in one epoch, it decreases with the number of epochs
                   mutation_factor=0.3,
                   # Current epoch
                   epoch=1,
                   # Minimum amount of mutation for each epoch
                   min_mutation=0.002,
                   # Probability a weight can have a big mutation
                   big_mutation_probability=0.0001,
                   # How much a weight can mutate, when a big mutation happens
                   big_mutation_factor=3,
                   # K-factor, number of recommenders
                   k=2,
                   # Consistency range, how much a weight in {i} position can be more important with respect to the
                   # one in position {i - 1}
                   consistency_range=0.08):
    # Copy the weights array
    weights = np.array(copy.deepcopy(weights))

    # Compute the parameters
    big_mutation_probability = big_mutation_probability / k
    big_mutation_factor = big_mutation_factor * k
    factor = mutation_factor / math.sqrt(epoch) + min_mutation

    # Mutate rows
    for row in range(len(weights)):
        mutation_amount = (1 - factor / 2) + rnd.random() * factor
        weights[row] = weights[row] * mutation_amount

    # Mutate each weight
    for row in range(len(weights)):
        for col in range(len(weights[row])):
            # If big mutation
            if rnd.random() < big_mutation_probability:
                # Increase
                if rnd.random() < 0.5:
                    weights[row][col] = weights[row][col] / big_mutation_factor
                # Decrease
                else:
                    weights[row][col] = weights[row][col] * big_mutation_factor
            # If standard mutation
            else:
                mutation_amount = (1 - factor / 2) + rnd.random() * factor
                weights[row][col] = weights[row][col] * mutation_amount
            # Consistency check:
            # The element on the left have always more weight
            if col > 0:
                weights[row][col] = min(weights[row][col], weights[row][col - 1] * (1 + consistency_range))

    # Normalize the data
    # Max value -> 1
    max_weight = weights.max()
    weights = weights / max_weight

    # Return the weights
    return weights


if __name__ == '__main__':


    # TODO: Edit here
    # Number of different weights to tune in each epoch
    population_size = 2
    # Number of epochs
    epochs = 5
    # Performances of the weights
    score = [None] * population_size
    # Number of different dataset to use
    anti_overfitting = 2
    # New people at each epoch
    # It should not be more than {population_size/2}
    new_population_at_each_epoch = 1
    # Save weights at each N epoch
    save_weights_every_N_epochs = 10
    # Parallelism (suggested: same as anti_overfitting, or a number that can divide the anti_overfitting value)
    parallelism = anti_overfitting
    # Anti variance parameter
    # The higher it is, the recommenders with less variance will survive
    anti_variance = 0.6
    # Max cutoff, items that each recommender has to recommend
    max_cutoff = 30
    # The file where the detailed scores are stored
    f_detailed_scores = open("potentissimo_details.info", "a+")
    # The file where the weights are stored
    f_weights = open("potentissimo_weights.info", "a+")
    # The file where the main info are stored
    f_main_info = open("potentissimo_info.info", "a+")

    # Auxiliary structures
    recommenders = np.full((population_size, anti_overfitting), None)
    base_recommenders = []
    tested_recommenders = []
    datas = [None] * anti_overfitting
    tested_users = []
    rec_model = []

    for j in range(anti_overfitting):
        data_reader = DataReader()
        datas[j] = DataObject(data_reader, 1, random_seed=(50 + j * 10))

        # TODO: Edit here
        # Insert the mix of recommender systems
        rec1 = ItemKNNCFRecommender(datas[j].urm_train)
        rec1.fit(shrink=10, topK=1000, feature_weighting="TF-IDF")
        rec2 = RP3betaRecommender(datas[j].urm_train)
        rec2.fit(topK=7, alpha=0.25, beta=0.09, implicit=True)
        rec3 = UserKNNCFRecommender(datas[j].urm_train)
        rec3.fit(topK=2000, shrink=10, similarity="jaccard", feature_weighting="BM25")
        # rec4 = ItemKNNCFRecommender(datas[j].urm_train)
        # rec4.fit(shrink=30, topK=30, feature_weighting="none", similarity="tanimoto")
        base_recommenders.append([rec1, rec2, rec3])

        # TODO: Edit here
        # Insert the type of users to test the recommenders
        n, t_users, description = datas[j].urm_train_users_by_type[8]

        tested_users.append(t_users)

        # TODO: Edit here
        # This code auto assign the weights
        # It is possible to manually assign the weights
        # For example, we can assign the weights of a previously tuned recommender
        cached_hybrid_rec = Hybrid1CXAlphaRecommender(datas[j], base_recommenders[j], tested_users[j], max_cutoff=30)
        # Edit here the weight if necessary
        # The size of the weights matrix must be bigger of the cutoff
        # cached_hybrid_rec.weights = []


        rec_model.append(cached_hybrid_rec)


    k_factor = len(base_recommenders[0])

    for i in range(population_size):
        for j in range(anti_overfitting):
            rec = rec_model[j].clone()
            recommenders[i][j] = rec



    #Weight mutation
    for i in range(population_size):
        ws = mutate_weights(recommenders[i][0].weights, k=k_factor)
        for j in range(anti_overfitting):
            recommenders[i][j].weights = ws

    # Time monitoring
    start_time = time.time()

    # Start the computation
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        f_main_info.write(f"epoch {epoch}\n")
        f_weights.write(f"epoch {epoch}\n")
        f_detailed_scores.write(f"epoch {epoch}\n")
        print("Epoch {} of {} complete in {:.2f} minutes\n".format(epoch, epochs,
                                                                   float(time.time() - start_time) / 60))
        f_main_info.write("Epoch {} of {} complete in {:.2f} minutes\n".format(epoch, epochs,
                                                                     float(time.time() - start_time) / 60))
        f_main_info.flush()
        f_weights.flush()
        f_detailed_scores.flush()
        start_time = time.time()

        for i in range(population_size):

            # Defining the function that will be parallelized
            def parallel_run(j):
                _rec = recommenders[i][j]
                _tested_users = tested_users[j]
                _result_string, _map = MyEvaluator.evaluate_algorithm(datas[j].urm_test, _tested_users, _rec)
                return _map

            partial_scores = Parallel(n_jobs=parallelism)(
                delayed(parallel_run)(generation) for generation in range(anti_overfitting))

            # Compute the score for each recommender
            # The scores computed are : average_map, stdev and the score
            average_score = 0
            for s in partial_scores:
                average_score += s
            average_score = average_score / anti_overfitting
            print(f"\t\taverage_map : {average_score}")
            f_detailed_scores.write(f"\t\taverage_map : {average_score}\n")
            standard_deviation = 0
            for s in partial_scores:
                standard_deviation += (s - average_score) * (s - average_score)
            standard_deviation = math.sqrt(standard_deviation / (anti_overfitting - 1))
            print(f"\t\tstandard_deviation : {standard_deviation}")
            f_detailed_scores.write(f"\t\tstandard_deviation : {standard_deviation}\n")
            average_score = average_score - standard_deviation * anti_variance
            print(f"\taverage_score : {average_score}")
            f_detailed_scores.write(f"\taverage_score : {average_score}\n")
            f_detailed_scores.flush()
            score[int(i)] = (average_score, recommenders[i])

        # Sort the recommenders by their average scores
        sorted_scores = sorted(score, key=operator.itemgetter(0), reverse=True)
        print(f"max_score : {sorted_scores[0][0]}")
        f_main_info.write(f"max_score : {sorted_scores[0][0]}\n")
        # Take the best ones
        best_recs = [t[1] for t in sorted_scores]

        # Save the weights
        if epoch % save_weights_every_N_epochs == 0 or epoch == epochs - 1:
            for i in range(int(population_size)):
                for w in recommenders[i][0].weights:
                    f_weights.write(f"w {i}\n -> {w.tolist()} \n")
                    f_weights.flush()

        # If the epoch is not the last one
        if epoch != (epochs - 1):
            # Replace the worst recommenders with an evolution of the best ones
            for i in range(new_population_at_each_epoch):
                ws = mutate_weights(best_recs[i][0].weights, epoch=epoch + 1, k=k_factor)
                new_ws = mutate_weights(ws, epoch=epoch + 1, k=k_factor)
                for j in range(anti_overfitting):
                    recommenders[i][j] = best_recs[i][j].clone()
                    recommenders[i][j].weights = copy.deepcopy(ws)
                    new_rec = best_recs[i][j].clone()
                    recommenders[-(i+1)][j] = new_rec
                    recommenders[-(i+1)][j].weights = copy.deepcopy(ws)
            for i in range(new_population_at_each_epoch, population_size - new_population_at_each_epoch):
                ws = mutate_weights(best_recs[i][0].weights, epoch=epoch + 1, k=k_factor)
                new_ws = mutate_weights(ws, epoch=epoch + 1, k=k_factor)
                for j in range(anti_overfitting):
                    recommenders[i][j] = best_recs[i][j].clone()
                    recommenders[i][j].weights = copy.deepcopy(ws)

