import numpy as np
import pandas as pd

from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


class MainRecommender:
    
    def __init__(self, data, weighting=True):
        
        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]
        
        # Топ покупок по всему датасету
        self.overall_top_perchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_perchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_perchases = self.overall_top_perchases[self.overall_top_perchases['item_id'] != 999999]
        self.overall_top_perchases = self.overall_top_perchases.item_id.tolist() 
        
        
        self.popularity = self.get_popularity(data) 
        
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)   
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T    
            
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        
        
    @staticmethod
    def get_popularity(data):
        
        popularity = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)
        
        return popularity    
         
    @staticmethod
    def prepare_matrix(data):
        """Готовит user-item матрицу"""
        
        data.columns = [col.lower() for col in data.columns]
        
        user_item_matrix = pd.pivot_table(data, 
                                  index='user_id', columns='item_id', 
                                  values='quantity', # Можно пробовать другие варианты
                                  aggfunc='count', 
                                  fill_value=0
                                 )
        
        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        # model.fit(csr_matrix(self.user_item_matrix).T.tocsr())
        
        return model
    
    # def get_als_bm25_recommendations(self, user, N=5):
    #     res = [self.id_to_itemid[rec[0]] for rec in self.model.recommend(userid=self.userid_to_id[user], # userid - id от 0 до N
    #                        user_items=csr_matrix(self.user_item_matrix).tocsr(),   # на вход user-item matrix
    #                        N=N, # кол-во рекомендаций 
    #                        filter_already_liked_items=False, 
    #                        recalculate_user=False)]
    #     return res

    def update_dict(self, user_id):
        """Обновление словарей при появлении нового user / item"""
        
        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.value()))
            max_id += 1
            
            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})
            
    
    def get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)
        # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0] # И берем второй (не товар из аргумента метода)
        
        return self.id_to_itemid[top_rec]
    
    
    def extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""
        
        if len(recommendations) < N:
            recommendations.extend(self.overall_top_perchases[:N])
            recommendations = recommendations[:N]
            
            return recommendations
        
    # def get_own_recommendations(self, user, N=5):
    def get_own_recommendations(self, user, N):
        """Рекомендуем товары среди тех, которые юзер уже купил"""
        
        self.update_dict(user_id=user)
        return self.get_recommendations(user, model=self.own_recommender, N=N)
        
        
    def get_recommendations(user, model, sparse_user_item, N=5):
        """Рекомендуем топ-N товаров"""
        
        self.update_dict(user_id=user)
        res = [id_to_itemid[rec] for rec in 
                    model.recommend(userid=userid_to_id[user], 
                                    user_items=sparse_user_item,   # на вход user-item matrix
                                    N=N, 
                                    filter_already_liked_items=False, 
                                    filter_items=[itemid_to_id[999999]], 
                                    recalculate_user=True)[0]]
        
        res = self._extend_with_top_popular(res, N=N)
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res    

#     def get_recommendations(self, user, model, N=5):
        
#         """Рекомендации через стардартные библиотеки implicit"""

#         self.update_dict(user_id=user)
#         res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
#                                         user_items=csr_matrix(self.user_item_matrix).tocsr(),
#                                         N=N,
#                                         filter_already_liked_items=False,
#                                         filter_items=[self.itemid_to_id[999999]],
#                                         recalculate_user=True)]

#         res = self._extend_with_top_popular(res, N=N)

#         assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
#         return res
        
    
    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        
        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)
        
        res = top_users_purchases['item_id'].apply(lambda x: self.get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

        
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        
        res = []
        
        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N+1)
        similar_users = [rec[0] for rec in similar_users]
        similar_users = similar_users[1:] # удалим юзера из запроса
        
        for user in similar_users:
            res.extend(self.get_own_recommendations(user, N=1))
            
        res = self._extend_with_top_popular(res, N=N)
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    
    def recs_top_n(self, N=5):
        res = self.popularity.groupby('item_id')['quantity'].count().reset_index()
        res.sort_values('quantity', ascending=False, inplace=True)
        res = res.item_id.head(N).tolist()
        
        return res
    