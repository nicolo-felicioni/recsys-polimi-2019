import operator
import copy
import math
import random
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
                   mutation_factor=0.4,
                   # Current epoch
                   epoch=1,
                   # Minimum amount of mutation for each epoch
                   min_mutation=0.005,
                   # Probability a weight can have a big mutation
                   big_mutation_probability=0.005,
                   # How much a weight can mutate, when a big mutation happens
                   big_mutation_factor=1.5,
                   # K-factor, number of recommenders
                   k=2,
                   # Consistency range, how much a weight in {i} position can be more important with respect to the
                   # one in position {i - 1}
                   consistency_range=0.15):
    # Copy the weights array
    weights = np.array(copy.deepcopy(weights))

    # Compute the parameters
    big_mutation_probability = big_mutation_probability / k
    big_mutation_factor = big_mutation_factor
    factor = mutation_factor / math.sqrt(epoch) + min_mutation

    # Mutate rows
    for row in range(len(weights)):
        mutation_amount = (1 - factor) + rnd.random() * factor
        weights[row] = weights[row] * mutation_amount

    # Mutate each weight
    for row in range(len(weights)):
        for col in range(len(weights[row])):
            # If big mutation
            if rnd.random() < big_mutation_probability:
                # Increase
                if rnd.random() < 0.05:
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


# Defining the function that will be parallelized
def parallel_run(j, rec, test_urm, tested_users):
    _rec = rec
    _tested_users = tested_users
    _test_urm = test_urm
    _result_string, _map = MyEvaluator.evaluate_algorithm(_test_urm, _tested_users, _rec)
    return _map



if __name__ == '__main__':


    # TODO: Edit here
    # Number of different weights to tune in each epoch
    population_size = 10
    # Number of epochs
    epochs = 30
    # Performances of the weights
    score = [None] * population_size
    # Number of different dataset to use
    anti_overfitting = 6
    # New people at each epoch
    # It should not be more than {population_size/2}
    new_population_at_each_epoch = 2
    # Save weights at each N epoch
    save_weights_every_N_epochs = 5
    # Parallelism (suggested: same as anti_overfitting, or a number that can divide the anti_overfitting value)
    parallelism = anti_overfitting
    # Anti variance parameter
    # The higher it is, the recommenders with less variance will survive
    anti_variance = 0.7
    # Max cutoff, items that each recommender has to recommend
    max_cutoff = 30
    # Number of new dataset generated
    # Be careful, it is a time consuming operation
    N_new_dataset = 1
    # Number of epochs to wait before generating new datasets
    # Be careful, it is a time consuming operation
    new_dataset_every_N_epoch = 3
    # Boosting the new weights by looking at why the worst ones suck
    opposing_factor = 0.3
    # Directory where the files are saved
    path = "001_more_tuning/"
    # The file where the detailed scores are stored
    f_detailed_scores = open(path + "potentissimo_details.info", "a+")
    # The file where the weights are stored
    f_weights = open(path + "potentissimo_weights.info", "a+")
    # The file where the main info are stored
    f_main_info = open(path + "potentissimo_info.info", "a+")
    # Dataset to change at each epoch, decreases overfitting
    # The file where the best weights are stored
    f_best_weights = open(path + "potentissimo_best_weights.info", "a+")

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
        # TODO: If you edit here, remember to edit at the end too
        # Insert the mix of recommender systems
        rec1 = UserKNNCFRecommender(datas[j].urm_train)
        rec1.fit(shrink=1000, topK=1000, similarity="cosine", feature_weighting="TF-IDF")
        rec2 = RP3betaRecommender(datas[j].urm_train)
        rec2.fit(topK=5000, alpha=0.35, beta=0.025, implicit=True)
        rec3 = ItemKNNCFRecommender(datas[j].urm_train)
        rec3.fit(topK=200, shrink=1000, similarity="jaccard", feature_weighting="TF-IDF")
        rec4 = UserKNNCBFRecommender(datas[j].ucm_all, datas[j].urm_train)
        rec4.fit(topK=5000, shrink=5, feature_weighting="TF-IDF", similarity="euclidean")
        rec5 = ItemKNNCBFRecommender(datas[j].urm_train, datas[j].icm_all_augmented)
        rec5.fit(topK=100, shrink=1, feature_weighting="TF-IDF", similarity="cosine")
        base_recommenders.append([rec1, rec2, rec3, rec4, rec5])

        # TODO: Edit here
        # Insert the type of users to test the recommenders
        n, t_users, description = datas[j].urm_train_users_by_type[1]

        tested_users.append(t_users)

        # TODO: Edit here
        # This code auto assign the weights
        # It is possible to manually assign the weights
        # For example, we can assign the weights of a previously tuned recommender
        cached_hybrid_rec = Hybrid1CXAlphaRecommender(datas[j], base_recommenders[j], tested_users[j], max_cutoff=max_cutoff)
        # Edit here the weight if necessary
        # The size of the weights matrix must be bigger of the cutoff
        cached_hybrid_rec.weights = np.array([
            [1.0, 0.7803233193098372, 0.4042650517804784, 0.26444366709538053, 0.2752283977870088, 0.23751321242187357,
             0.24058696112550104, 0.23590434442820032, 0.1992990685621503, 0.1878790301115923, 0.16690930495160153,
             0.15325757938895435, 0.1655181857400707, 0.16904924923310471, 0.13097329147255055, 0.12288596504305309,
             0.11857054225592471, 0.1219683401906944, 0.12310761424074275, 0.12706253865684247, 0.10616907384825704,
             0.11435056385016795, 0.10762546479680782, 0.10853202794520854, 0.10581279995920735, 0.030542092731989226,
             0.02815866621379197, 0.029606210545982038, 0.030081711307335356, 0.021115678446098837],
            [0.5617369140815882, 0.5051503534413744, 0.4122970541828909, 0.3042070282190491, 0.17304957495333953,
             0.15975399321439848, 0.14814728006524608, 0.15134476963342136, 0.1542220148819716, 0.11561331022562493,
             0.10908477814827285, 0.10878344187833902, 0.1076874074506557, 0.10060130168212467, 0.10121643145473896,
             0.1051672550351394, 0.10914214926956825, 0.11561944751049083, 0.09503842722768421, 0.10195954373713821,
             0.08389274900027614, 0.07778508607315454, 0.07202046163295679, 0.05037543556069969, 0.05151145723008148,
             0.05209253455852665, 0.05625993732320878, 0.045214650939956184, 0.010633151697055027, 0.0],
            [0.7970208723559576, 0.26588843253273464, 0.27451972873900987, 0.2594629845684418, 0.2704505019435646,
             0.28847200221236347, 0.29769381061823086, 0.3033002624992306, 0.2858666306724928, 0.10873408621663458,
             0.1145941069878546, 0.1007352153416693, 0.09577285319339945, 0.10343468144887141, 0.10718040282486481,
             0.10628112303689995, 0.10864553230333267, 0.11733717488759929, 0.1235943884512356, 0.12977936662545492,
             0.1328216893124467, 0.0852451467531702, 0.07284498682886342, 0.07292873248169329, 0.06482031742769725,
             0.059949614092690624, 0.048801873413420856, 0.04449798601968061, 0.014585128187505443, 0.0],
            [0.312709722208491, 0.1679689986131832, 0.09574967664805611, 0.0937168516142597, 0.06730472815238422,
             0.05793939858645846, 0.06190923808499407, 0.06161403716841971, 0.061253553363129, 0.06034606218912711,
             0.0623793043883789, 0.056165578039633186, 0.05954068051677049, 0.05732010154404945, 0.04691492897063987,
             0.045855446021107304, 0.0468354260939584, 0.04439236189785672, 0.040409036144857205, 0.04220314843944912,
             0.04548928016201977, 0.04771764051133145, 0.038304114258795575, 0.023510374098401236, 0.023735450774252025,
             0.015672456573763406, 0.012687787761657573, 0.011696991089728612, 0.0, 0.0],
            [0.5675832440295991, 0.5139782709409084, 0.4316250124743416, 0.4568530018138936, 0.4919703838474551,
             0.5179189119368128, 0.45971139553064755, 0.3773984935335688, 0.38040216226663587, 0.3100840107916866,
             0.29255136126951053, 0.2998429110213814, 0.1777264406266002, 0.18402808789278235, 0.14939740662135373,
             0.15533587083320555, 0.15878558194050288, 0.17148842849574314, 0.1562252926917673, 0.1687233161071087,
             0.1683264883084783, 0.14832095865542383, 0.15825311090198418, 0.08862126192009957, 0.08340587619565315,
             0.025133585390153763, 0.02635202712950645, 0.026891026097292218, 0.028544531720080774, 0.0]
        ])


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
        f_best_weights.write(f"epoch {epoch}\n")
        print("Epoch {} of {} complete in {:.2f} minutes\n".format(epoch, epochs,
                                                                   float(time.time() - start_time) / 60))
        f_main_info.write("Epoch {} of {} complete in {:.2f} minutes\n".format(epoch, epochs,
                                                                     float(time.time() - start_time) / 60))
        f_main_info.flush()
        f_weights.flush()
        f_best_weights.flush()
        f_detailed_scores.flush()
        start_time = time.time()

        for i in range(population_size):

            # Defining the function that will be parallelized
            # def parallel_run(j):
            #     _rec = recommenders[i][j]
            #     _tested_users = tested_users[j]
            #     _result_string, _map = MyEvaluator.evaluate_algorithm(datas[j].urm_test, _tested_users, _rec)
            #     return _map
            partial_scores = Parallel(n_jobs=parallelism)(
                delayed(parallel_run)(generation, recommenders[i][generation], datas[generation].urm_test, tested_users[generation]) for generation in range(anti_overfitting))

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
        # Take f_best_weights best ones
        best_recs = [t[1] for t in sorted_scores]
        print(f"max_score : {sorted_scores[0][0]}")
        f_main_info.write(f"max_score : {sorted_scores[0][0]}\n")
        f_best_weights.write(f"best_weights :\n")
        for w in best_recs[0][0].weights:
            f_best_weights.write(f"w {i}\n -> {w.tolist()} \n")
            f_weights.flush()

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
                # Opposing boost
                # delta_ws = (ws - best_recs[-(i+1)][0].weights) * opposing_factor
                new_ws = mutate_weights(best_recs[i][0].weights, epoch=epoch + 1, k=k_factor)
                for j in range(anti_overfitting):
                    recommenders[i][j] = best_recs[i][j].clone()
                    recommenders[i][j].weights = copy.deepcopy(ws)
                    new_rec = best_recs[i][j].clone()
                    recommenders[-(i+1)][j] = new_rec
                    recommenders[-(i+1)][j].weights = copy.deepcopy(new_ws)
            for i in range(new_population_at_each_epoch, population_size - new_population_at_each_epoch):
                ws = mutate_weights(best_recs[i][0].weights, epoch=epoch + 1, k=k_factor)
                new_ws = mutate_weights(ws, epoch=epoch + 1, k=k_factor)
                for j in range(anti_overfitting):
                    recommenders[i][j] = best_recs[i][j].clone()
                    recommenders[i][j].weights = copy.deepcopy(ws)

            if epoch % new_dataset_every_N_epoch == (new_dataset_every_N_epoch - 1):
                for _ in range(N_new_dataset):
                    random_dataset_index = random.randint(0, anti_overfitting) - 1
                    data_reader = DataReader()
                    j = random_dataset_index
                    datas[j] = DataObject(data_reader, 1, random_seed=(random.randint(0, 200)))


                    # TODO: Edit here, if you changed the base recommenders
                    rec1 = UserKNNCFRecommender(datas[j].urm_train)
                    rec1.fit(shrink=1000, topK=1000, similarity="cosine", feature_weighting="TF-IDF")
                    rec2 = RP3betaRecommender(datas[j].urm_train)
                    rec2.fit(topK=5000, alpha=0.35, beta=0.025, implicit=True)
                    rec3 = ItemKNNCFRecommender(datas[j].urm_train)
                    rec3.fit(topK=200, shrink=1000, similarity="jaccard", feature_weighting="TF-IDF")
                    rec4 = UserKNNCBFRecommender(datas[j].ucm_all, datas[j].urm_train)
                    rec4.fit(topK=5000, shrink=5, feature_weighting="TF-IDF", similarity="euclidean")
                    rec5 = ItemKNNCBFRecommender(datas[j].urm_train, datas[j].icm_all_augmented)
                    rec5.fit(topK=100, shrink=1, feature_weighting="TF-IDF", similarity="cosine")
                    # TODO: edit also here
                    n, t_users, description = datas[j].urm_train_users_by_type[1]
                    base_recommenders = [rec1, rec2, rec3, rec4, rec5]


                    tested_users.append(t_users)
                    cached_hybrid_rec = Hybrid1CXAlphaRecommender(datas[j], base_recommenders, tested_users[j], max_cutoff=max_cutoff)
                    for i in range(population_size):
                        recommenders[i][j].cached_recommendation_all = cached_hybrid_rec.cached_recommendation_all

