from functools import partial

from scipy.optimize import OptimizeResult
from skopt import gp_minimize, forest_minimize
from skopt.space import *

from Base.Evaluation import MyEvaluator
from Custom.Potentissimo_2Rec import Potentissimo_2Rec
from DataObject import DataObject
from DataReader import DataReader
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Utils.weights_generator import *


def gen_dataset(seed):
    random_seed = seed
    data_reader = DataReader()
    return DataObject(data_reader, 1, random_seed=random_seed)


class Eval_2:

    def __init__(self, hybrid_recs, target_user_list, dataset_list):
        self.hybrid_recs = hybrid_recs
        self.target_user_list = target_user_list
        self.dataset_list = dataset_list
        self.counter = 0

    def eval(self, *args):
        distribution_1 = args[0][0]
        mean_1 = args[0][1]
        offset_1 = args[0][2]
        base_1 = args[0][3]
        size_1 = args[0][4]
        distribution_2 = args[0][5]
        mean_2 = args[0][6]
        offset_2 = args[0][7]
        base_2 = args[0][8]
        size_2 = args[0][9]

        _hybrid_recs = self.hybrid_recs
        _target_user_list = self.target_user_list
        _dataset_list = self.dataset_list
        weights_1 = []
        if distribution_1 == 'LIN':
            weights_1 = linear(mean=mean_1, offset=offset_1, size=size_1)
        elif distribution_1 == 'EXP':
            weights_1 = exponential(mean=mean_1, offset=offset_1, size=size_1, base=base_1)
        elif distribution_1 == 'LOG':
            weights_1 = logarithmic(mean=mean_1, offset=offset_1, size=size_1, base=base_1)

        weights_2 = []
        if distribution_2 == 'LIN':
            weights_2 = linear(mean=mean_2, offset=offset_2, size=size_2)
        elif distribution_2 == 'EXP':
            weights_2 = exponential(mean=mean_2, offset=offset_2, size=size_2, base=base_2)
        elif distribution_2 == 'LOG':
            weights_2 = logarithmic(mean=mean_2, offset=offset_2, size=size_2, base=base_2)

        for rec in hybrid_recs:
            rec.fit(weights_1, weights_2)

        records = zip(_dataset_list, _target_user_list, _hybrid_recs)
        map_result = [MyEvaluator.evaluate_algorithm(data.urm_test, target_user, rec)[1] for data, target_user, rec in
                      records]

        # Negate map
        map_result = np.array(map_result).mean() * -1

        self.counter += 1
        print(f"run # {self.counter}")
        print(f"parameters # {args}")
        print(f"map # {map_result}\n\n")

        return map_result


if __name__ == '__main__':
    n_dataset = 4
    max_cutoff = 30

    dataset_list = [gen_dataset(x) for x in range(n_dataset)]
    recs_1 = [RP3betaRecommender(data.urm_train) for data in dataset_list]
    [rec.fit(topK=5000, alpha=0.35, beta=0.025, implicit=True)
     for rec in recs_1]
    recs_2 = [ItemKNNCFRecommender(data.urm_train) for data in dataset_list]
    [rec.fit(topK=200, shrink=1000, similarity="jaccard", feature_weighting="TF-IDF")
     for rec in recs_2]
    target_user_list = [data.urm_train_users_by_type[1][1] for data in dataset_list]
    records = zip(dataset_list, recs_1, recs_2, target_user_list)
    hybrid_recs = [Potentissimo_2Rec(data=dataset,
                                     rec1=rec1,
                                     rec2=rec2,
                                     target_users=target_user,
                                     max_cutoff=max_cutoff) for dataset, rec1, rec2, target_user in records]
    eval = Eval_2(hybrid_recs=hybrid_recs, target_user_list=target_user_list, dataset_list=dataset_list)

    hyperparameters = [
        Categorical(['LIN', 'EXP', 'LOG'], name="distribution_1"),
        Real(1, 2, name="mean_1"),
        Real(0, 1, name="offset_1"),
        Real(0.5, 2, name="base_1"),
        Integer(5, max_cutoff, name="size_1"),
        Categorical(['LIN', 'EXP', 'LOG'], name="distribution_2"),
        Real(1, 2, name="mean_2"),
        Real(0, 1, name="offset_2"),
        Real(0.5, 2, name="base_2"),
        Integer(5, max_cutoff, name="size_2")
    ]

    # res = gp_minimize(eval.eval,  # the function to minimize
    #                   hyperparameters,  # the bounds on each dimension of x
    #                   acq_func="PI",  # the acquisition function
    #                   acq_optimizer="auto",  # the acquisition function
    #                   n_calls=20,  # the number of evaluations of f
    #                   n_random_starts=1,  # the number of random initialization points
    #                   noise=0.002,  # the noise level (optional)
    #                   random_state=123,
    #                   verbose=False)  # the random seed

    res = forest_minimize(eval.eval,  # the function to minimize
                          hyperparameters,
                          n_calls=500,
                          n_random_starts=50,
                          n_jobs=4)
