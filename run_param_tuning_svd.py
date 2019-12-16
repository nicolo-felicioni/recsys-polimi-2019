from joblib import Parallel, delayed
from skopt import gp_minimize, forest_minimize, gbrt_minimize, load, dump
import numpy as np
from skopt.space import Integer, Categorical, Real
import time
from Base.Evaluation import MyEvaluator
from Base.Evaluation.Evaluator import EvaluatorHoldout
from DataObject import DataObject
from DataReader import DataReader
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython, \
    MatrixFactorization_AsySVD_Cython


def gen_dataset(seed):
    random_seed = seed
    data_reader = DataReader()
    return DataObject(data_reader, 1, random_seed=random_seed)


def parallel_fit_and_eval_job(recommender, data: DataObject, epochs, num_factors, learning_rate, sgd,
                              negative_interactions_quota, init_mean, init_std_dev,
                              user_reg, item_reg, bias_reg, positive_reg, negative_reg):
    ev = EvaluatorHoldout(data.urm_test, [10], minRatingsPerUser=1, exclude_seen=True,
                          verbose=True)

    # Fit
    recommender.fit(epochs=epochs, batch_size=1000,
                    num_factors=num_factors, positive_threshold_BPR=None,
                    learning_rate=learning_rate, use_bias=True,
                    sgd_mode=sgd,
                    negative_interactions_quota=negative_interactions_quota,
                    init_mean=init_mean, init_std_dev=init_std_dev,
                    user_reg=user_reg, item_reg=item_reg, bias_reg=bias_reg, positive_reg=positive_reg,
                    negative_reg=negative_reg,
                    validation_every_n=3,
                    epochs_min=1,
                    stop_on_validation=True,
                    validation_metric="MAP",
                    lower_validations_allowed=2,
                    evaluator_object=ev,
                    random_seed=None)

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
                 target_user_list,
                 dataset_list,
                 type_of_user,
                 parallelism=2,
                 filename_csv="skopt_run.csv"):
        self.target_user_list = target_user_list
        self.dataset_list = dataset_list
        self.type_of_user = type_of_user
        self.parallelism = parallelism
        self.filename_csv = filename_csv
        self.counter = 0
        self.timer = time.time()

    def eval(self, *args):
        # Input parameters
        epochs = args[0][0]
        num_factors = args[0][1]
        learning_rate = args[0][2]
        sgd = args[0][3]
        negative_interactions_quota = args[0][4]
        init_mean = args[0][5]
        init_std_dev = args[0][6]
        user_reg = args[0][7]
        item_reg = args[0][8]
        bias_reg = args[0][9]
        positive_reg = args[0][10]
        negative_reg = args[0][11]

        # Text used for csv file
        input_as_string = f"epochs={epochs} - num_factors={num_factors} - learning_rate={learning_rate} - sgd={sgd} - " \
                          f"negative_interactions_quota={negative_interactions_quota} - init_mean={init_mean} - " \
                          f"init_std_dev={init_std_dev} - user_reg={user_reg} - item_reg={item_reg} - " \
                          f"bias_reg={bias_reg} - positive_reg={positive_reg} - negative_reg={negative_reg}"
        recommender_name = "AsySVD"

        # Creating the recommenders (parallel fit and evaluation)
        recs = [MatrixFactorization_AsySVD_Cython(data.urm_train) for data in self.dataset_list]
        pairs = zip(recs, self.dataset_list)
        results = Parallel(n_jobs=parallelism)(
            delayed
            (parallel_fit_and_eval_job)
            (rec, data, epochs, num_factors, learning_rate, sgd, negative_interactions_quota, init_mean, init_std_dev,
             user_reg, item_reg, bias_reg, positive_reg, negative_reg)
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
    type_of_user = 0  # Type of the user to evaluate
    parallelism = 1  # Number of thread used for training and evaluation
    n_load_and_rerun = 1

    # Skopt configuration
    acq_func = "EI"  # The acquisition function
    acq_optimizer = "auto"
    base_estimator = "RF"  # It can be "RF" (Random Forest) or "ET" (Extra Tree). The first one is more time consuming.
    n_calls = 20
    n_random_starts = 2
    random_state = 100
    n_jobs = 4

    # Persistence configuration
    filename_skopt = "asysvd_5.pkl"
    filename_csv = "asysvd.csv"

    dataset_list = Parallel(n_jobs=parallelism)(
        delayed
        (gen_dataset)
        (x + 20)
        for x in range(n_dataset))
    target_user_list = [data.urm_train_users_by_type[type_of_user][1] for data in dataset_list]

    records = zip(dataset_list, target_user_list)

    eval = Evaluator(target_user_list=target_user_list,
                     dataset_list=dataset_list,
                     type_of_user=type_of_user,
                     parallelism=parallelism,
                     filename_csv=filename_csv)

    hyperparameters = [
        Integer(10, 1000, name="epochs"),
        Integer(5, 100, name="num_factors"),
        Real(0, 1, name="learning_rate"),
        Categorical(["sgd", "adam", "adagrad", "rmsprop"], name="sgd"),
        Real(0, 1, name="negative_interactions_quota"),
        Real(0, 2, name="init_mean"),
        Real(0, 2, name="init_std_dev"),
        Real(0, 2, name="user_reg"),
        Real(0, 2, name="item_reg"),
        Real(0, 2, name="bias_reg"),
        Real(0, 2, name="positive_reg"),
        Real(0, 2, name="negative_reg")
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
