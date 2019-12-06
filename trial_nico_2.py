import time

from Base.Evaluation import MyEvaluator
from DataObject import DataObject
from DataReader import DataReader
import pandas as pd
from Hybrid.Hybrid003AlphaRecommender import Hybrid003AlphaRecommender
from Hybrid.Hybrid108AlphaRecommender import Hybrid108AlphaRecommender
from Hybrid.HybridG00AlphaRecommender import HybridG00AlphaRecommender
from Hybrid.WrapperThres import WrapperThres
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.NewUserKNNAgeRecommender import NewUserKNNAgeRecommender
from KNN.UserKNNAgeRecommender import UserKNNAgeRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython
from MatrixFactorization.IALSRecommender import IALSRecommender
from MatrixFactorization.NMFRecommender import NMFRecommender
from MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from GraphBased.RP3betaRecommender import RP3betaRecommender

def total_evaluation(rec):
    for n, users, description in data.urm_train_users_by_type:
        eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec, at=10, remove_top=0)
        print(f"\t {description},\t {eval}")
    print(MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_warm_user, rec))


def make_submission():
    data_reader_loc = DataReader()
    data_loc = DataObject(data_reader_loc, k=0, random_seed=1234)
    data_loc.urm_train = data_loc.urm
    recommender = Hybrid003AlphaRecommender(data_loc)
    recommender.fit()
    f = open("submission_pot_rp3.csv", "w+")
    f.write("user_id,item_list\n")
    for user_id in data_loc.ids_target_users:
        recommended_items = recommender.recommend(user_id, cutoff=10)
        print(user_id)
        print(recommended_items)
        well_formatted = " ".join([str(x) for x in recommended_items])
        f.write(f"{user_id}, {well_formatted}\n")

def trial_boost_user(data):
    rec1 = UserKNNCFRecommender(data.urm_train)
    rec2 = NewUserKNNAgeRecommender(data)
    rec3 = UserKNNCFRecommender(data.urm_train)
    rec1.fit(topK=500, shrink=1000, similarity='cosine', feature_weighting='TF-IDF')
    rec2.set_similarity([1])
    rec2.fit(topK=0)

    for alpha in [0.999990, 0.999995, 0.999999]:
        print("alpha:" + str(alpha))
        rec3.W_sparse = alpha * rec1.W_sparse + (1 - alpha) * rec2.W_sparse
        total_evaluation(rec3)





def trial_agerec(data):
    ucm_age_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_UCM_age.csv"
    df_age = pd.read_csv(ucm_age_path)
    # rec1 = UserKNNAgeRecommender(data, df_age)
    # rec1.fit(topK=1)
    rec = UserKNNAgeRecommender(data, df_age)
    rec.fit(topK=1)
    for n, users, description in data.urm_train_users_by_type:
        eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec, at=10, remove_top=0)
        print(f"\t {description},\t {eval}")
    print(MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_warm_user, rec))
    # print("two matrices are the same?")
    # ind = (rec1.W_sparse!=rec2.W_sparse).nnz
    # print(ind == 0)


# def create_ucm_interactions(data):
#     f = open("data_UCM_interaction.csv", "w+")
#     f.write("user_id,n_interactions\n")
#     for user_id in data.ids_user:
#         interactions = data.urm.getrow(user_id).getnnz()
#         well_formatted = str(interactions)
#         f.write(f"{user_id},{well_formatted},1.0\n")

def trial_inter(data):
    rec = UserKNNCBFRecommender(UCM=data.ucm_interaction, URM_train=data.urm_train)
    rec.fit()
    for n, users, description in data.urm_train_users_by_type:
        eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec, at=10, remove_top=0)
        print(f"\t {description},\t {eval}")
    print(MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_warm_user, rec))


if __name__ == "__main__":
    random_seed = 0
    data_reader = DataReader()
    data = DataObject(data_reader, k=1, random_seed=random_seed)
    print("seed: {}".format(random_seed))

    trial_boost_user(data)

    # rec = WrapperThres(data, threshold=69)
    # for n, users, description in data.urm_train_users_by_type:
    #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec, at=10, remove_top=0)
    #     print(f"\t {description},\t {eval}")
    # print(MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_warm_user, rec))


    # trial_inter(data)


    # make_submission()


    # trial_agerec(data)


    # ucm_age_path = "Data_manager_split_datasets/RecSys2019/recommender-system-2019-challenge-polimi/data_UCM_age.csv"
    # df_age = pd.read_csv(ucm_age_path)
    # rec1 = UserKNNAgeRecommender(data, df_age)
    # rec1.fit(topK=1)
    # rec2 = NewUserKNNAgeRecommender(data, df_age)
    # rec2.fit(topK=1)
    #
    # print("two matrices are the same?")
    # ind = (rec1.W_sparse != rec2.W_sparse).nnz
    # print(ind == 0)
    # create_ucm_interactions(data)
    # rec = RP3betaRecommender(data.urm_train)
    # rec.fit(topK=20, alpha=0.12, beta=0.28)
    # for n, users, description in data.urm_train_users_by_type:
    #     eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec, at=10, remove_top=0)
    #     print(f"\t {description},\t {eval}")
    # print(MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_warm_user, rec))

