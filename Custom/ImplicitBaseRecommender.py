from Base.BaseRecommender import BaseRecommender
from DataObject import DataObject


class ImplicitBaseRecommender(BaseRecommender):

    def __init__(self, urm_train,factors=100, regularization=0.01, use_native=True,
                                                        use_cg=True, use_gpu=False, iterations=15,
                                                        calculate_training_loss=False, num_threads=0):
        super(ImplicitBaseRecommender, self).__init__(urm_train)


    def fit(self, show_progress=True):
        self.rec.fit(self.URM_train.T, show_progress=show_progress)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_not_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        # items_to_be_recommended = [x
        #                            for x in self.data.ids_item
        #                            if x not in self.data.urm_train[user_id_array].indices]
        list_tuples_item_score = self.rec.recommend(user_id_array, self.URM_train,
                                                    filter_already_liked_items=remove_seen_flag, N=cutoff,
                                                    filter_items=items_to_not_compute)

        if (return_scores):
            return list_tuples_item_score
        else:
            list_items = []
            for tuple in list_tuples_item_score:
                item = tuple[0]
                list_items.append(item)
            return list_items