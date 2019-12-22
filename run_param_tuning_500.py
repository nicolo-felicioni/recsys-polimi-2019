from joblib import Parallel, delayed
from skopt import gp_minimize, forest_minimize, gbrt_minimize, load, dump
import numpy as np
from skopt.space import Integer, Categorical, Real
import time
from Base.Evaluation import MyEvaluator
from DataObject import DataObject
from DataReader import DataReader
from Hybrid.Hybrid400AlphaRecommender import Hybrid400AlphaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender


def gen_dataset(seed):
    random_seed = seed
    data_reader = DataReader()
    return DataObject(data_reader, 1, random_seed=random_seed)


def parallel_fit_and_eval_job(recommender, data: DataObject):

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
        print(args)
        # Input parameters
        k = args[0][0]
        leave_k_out = args[0][1]
        threshold = args[0][2]
        probability = args[0][3]

        # Text used for csv file
        input_as_string = f"k={k} - leave_k_out={leave_k_out} - threshold={threshold} - probability={probability}"
        recommender_name = "400 RP3 topk=20 - alpha=0.16 - beta=0.24"

        # Creating the recommenders (parallel fit and evaluation)
        recs = [Hybrid400AlphaRecommender(data, k, leave_k_out, threshold, probability) for data in self.dataset_list]
        pairs = zip(recs, self.dataset_list)
        results = Parallel(n_jobs=parallelism)(
            delayed
            (parallel_fit_and_eval_job)
            (rec, data)
            for rec, data in pairs)

        # Computing the average MAP
        map_per_type = np.array(results).mean(axis=0)

        partial = np.array(results) - map_per_type
        partial = np.square(partial).sum(axis=0)
        std_per_type = partial * 0
        if len(self.dataset_list) - 1 > 0:
            std_per_type = np.array(partial / (len(self.dataset_list) - 1))

        # Storing the information on file
        f = open(self.filename_csv, "a+")
        map_as_string = " ".join([str(x) + "," for x in map_per_type])
        std_as_string = " ".join([str(x) + "," for x in std_per_type])
        f.write(f"{recommender_name}, {input_as_string}, {map_as_string}, standard, {std_as_string}\n")
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
    type_of_user = 12  # Type of the user to evaluate
    parallelism = 1  # Number of thread used for training and evaluation
    n_load_and_rerun = 5

    # Skopt configuration
    acq_func = "EI"  # The acquisition function
    acq_optimizer = "auto"
    base_estimator = "RF"  # It can be "RF" (Random Forest) or "ET" (Extra Tree). The first one is more time consuming.
    n_calls = 2
    n_random_starts = 1
    random_state = 100
    n_jobs = 4

    # Persistence configuration
    filename_skopt = "400_RP3_12.pkl"
    filename_csv = "400_RP3.csv"

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
        Integer(2, 7),
        Integer(0, 4),
        Integer(0, 30),
        Real(0, 0.75)
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
