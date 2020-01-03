from Base.BaseRecommender import BaseRecommender
from Custom.ImplicitBaseRecommender import ImplicitBaseRecommender
from DataObject import DataObject
from Hybrid.Hybrid1CXXAlphaRecommender import Hybrid1CXXAlphaRecommender
import numpy as np
import implicit


class ImplicitALSRecommender(ImplicitBaseRecommender):
    """ImplicitALSRecommender recommender"""

    RECOMMENDER_NAME = "ImplicitALSRecommender"

    def __init__(self, data : DataObject,factors=100, regularization=0.01, use_native=True,
                                                        use_cg=True, use_gpu=False, iterations=15,
                                                        calculate_training_loss=False, num_threads=0):
        super(ImplicitALSRecommender, self).__init__(data.urm_train)
        self.data = data
        self.rec = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization,
                                                        use_native=use_native, use_cg=use_cg, use_gpu=use_gpu,
                                                        iterations=iterations,
                                                        calculate_training_loss=calculate_training_loss,
                                                        num_threads=num_threads)


