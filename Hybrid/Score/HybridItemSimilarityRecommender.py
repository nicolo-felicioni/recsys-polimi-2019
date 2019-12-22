from Base.BaseRecommender import BaseRecommender
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
import scipy.sparse as sps
import numpy as np

class HybridItemSimilarityRecommender(BaseItemSimilarityMatrixRecommender):
    """ HybridItemSimilarityRecommender
    """

    RECOMMENDER_NAME = "HybridItemSimilarityRecommender"

    def __init__(self, URM_train, verbose=True):
        super(HybridItemSimilarityRecommender, self).__init__(URM_train, verbose=verbose)

    def fit(self,
            similarities,
            weights=None,
            topK=100,
            normalize_weights=True):

        # Initialize weights array if not already initialized
        if weights is None:
            weights = np.array([1 for _ in similarities])

        # Checking the input parameters are well formatted
        assert len(similarities) == len(weights)
        assert len(similarities) > 0

        # Cast weights to numpy array if it is not
        weights = np.array(weights, dtype=np.float)
        # Normalize the weights
        if normalize_weights:
            weights /= weights.max()

        # Create a list of pairs (similarity, weight)
        similarity_and_weight = zip(similarities, weights)

        # Initialize the result
        W_sparse = sps.csr_matrix(similarities[0].shape, dtype=np.float)

        # Compute the new Similarity matrix
        for similarity, weight in similarity_and_weight:
            W_sparse += (similarity * weight)

        self.W_sparse = similarityMatrixTopK(W_sparse, k=topK)
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
