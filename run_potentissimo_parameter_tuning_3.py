from joblib import Parallel, delayed
from skopt import gp_minimize, forest_minimize, gbrt_minimize, load, dump
import numpy as np
from skopt.space import Integer, Categorical
import time
from Base.Evaluation import MyEvaluator
from DataObject import DataObject
from DataReader import DataReader
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender


def gen_dataset(seed):
    random_seed = seed
    data_reader = DataReader()
    return DataObject(data_reader, 1, random_seed=random_seed)


def parallel_fit_and_eval_job(recommender, data: DataObject, topK, shrink, similarity, normalize, feature_weighting):
    # Fit
    recommender.fit(topK, shrink, similarity, normalize, feature_weighting)

    # Eval
    _result = []
    for n, users, description in data.urm_train_users_by_type:
        _eval, _map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
        _result.append(_map)
    users = data.ids_target_users
    _eval, _map = MyEvaluator.evaluate_algorithm(data.urm_test, users, recommender, at=10, remove_top=0)
    _result.append(_map)
    return _result


class Evaluator:

    def __init__(self,
                 dataset_list,
                 type_of_user,
                 parallelism=2,
                 filename_csv="skopt_run.csv"):
        self.dataset_list = dataset_list
        self.type_of_user = type_of_user
        self.parallelism = parallelism
        self.filename_csv = filename_csv
        self.counter = 0
        self.timer = time.time()

    def eval(self, *args):
        # Input parameters
        topK = args[0][0]
        shrink = args[0][1]
        similarity = args[0][2]
        normalize = args[0][3]
        feature_weighting = args[0][4]

        # Text used for csv file
        input_as_string = f"topK={topK} - shrink={shrink} - similarity={similarity} - normalize={normalize} - " \
                          f"feature_weighting={feature_weighting}"
        recommender_name = "Item CF"

        # Creating the recommenders (parallel fit and evaluation)
        recs = [UserKNNCFRecommender(data.urm_train) for data in self.dataset_list]
        pairs = zip(recs, self.dataset_list)
        results = Parallel(n_jobs=parallelism)(
            delayed
            (parallel_fit_and_eval_job)
            (rec, data, topK, shrink, similarity, normalize, feature_weighting)
            for rec, data in pairs)

        # Computing the average MAP
        map_per_type = np.array(results).mean(axis=0)

        # Storing the information on file
        f = open(self.filename_csv, "a+")
        map_as_string = " ".join([str(x) + "," for x in map_per_type])
        f.write(f"{recommender_name}, {input_as_string}, {map_as_string}\n")
        f.flush()
        f.close()

        # The MAP value that should be optimized
        optimized_map = map_per_type[self.type_of_user]

        # Printing stuffs
        current_time = time.time()
        print(f"run : {self.counter} - computed in {current_time - self.timer} seconds")
        print(f"\tparameters : {input_as_string}")
        print(f"\tmap : {optimized_map}\n")
        self.counter += 1
        self.timer = current_time

        return -optimized_map


if __name__ == '__main__':

    # Run configuration
    n_dataset = 1  # Number of datasets
    type_of_user = 5  # Type of the user to evaluate
    parallelism = 1  # Number of thread used for training and evaluation
    n_load_and_rerun = 5

    # Skopt configuration
    acq_func = "EI"  # The acquisition function
    acq_optimizer = "auto"
    base_estimator = "RF"  # It can be "RF" (Random Forest) or "ET" (Extra Tree). The first one is more time consuming.
    n_calls = 1
    n_random_starts = 1
    random_state = 100
    n_jobs = 4

    # Persistence configuration
    filename_skopt = "userCF_5.pkl"
    filename_csv = "userCF.csv"

    dataset_list = Parallel(n_jobs=parallelism)(
        delayed
        (gen_dataset)
        (x + 20)
        for x in range(n_dataset))

    eval = Evaluator(dataset_list=dataset_list,
                     type_of_user=type_of_user,
                     parallelism=parallelism,
                     filename_csv=filename_csv)

    hyperparameters = [
        Integer(5, 20000),
        Integer(0, 10000),
        Categorical(["cosine", "jaccard", "asymmetric", "dice", "tversky"]),
        Categorical([True, False]),
        Categorical(["none", "BM25", "TF-IDF"])
    ]

    for _ in range(n_load_and_rerun):

        try:
            with open(filename_skopt, "rb") as f:
                res_loaded = load(f)
                f.close()
            res = forest_minimize(eval.eval,  # the function to minimize
                                  hyperparameters,  # the bounds on each dimension of x
                                  acq_func=acq_func,  # the acquisition function
                                  # acq_optimizer=acq_optimizer,  # the acquisition function
                                  n_calls=n_calls,  # the number of evaluations of f
                                  n_random_starts=n_random_starts,  # the number of random initialization points
                                  base_estimator=base_estimator,  # random forest as estimator
                                  verbose=False,
                                  n_jobs=n_jobs,
                                  x0=res_loaded.x_iters,
                                  y0=res_loaded.func_vals)  # the random seed
            with open(filename_skopt, 'wb') as f:
                dump(res, filename=f, compress=9)
                f.close()
        except:
            res = forest_minimize(eval.eval,  # the function to minimize
                                  hyperparameters,  # the bounds on each dimension of x
                                  acq_func=acq_func,  # the acquisition function
                                  # acq_optimizer=acq_optimizer,  # the acquisition function
                                  n_calls=n_calls,  # the number of evaluations of f
                                  n_random_starts=n_random_starts,  # the number of random initialization points
                                  base_estimator=base_estimator,  # random forest as estimator
                                  verbose=False,
                                  n_jobs=n_jobs)  # the random seed
            with open(filename_skopt, 'wb') as f:
                dump(res, filename=f, compress=9)
                f.close()
