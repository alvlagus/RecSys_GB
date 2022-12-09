"""
Metrics

"""
import numpy as np

def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return (flags.sum() > 0) * 1

# def hit_rate_at_k(recommended_list, bought_list, k=5):
#     return hit_rate(recommended_list[:k], bought_list)


def hit_rate_at_k(recommended_list, bought_list, k=5): 
    bought_list = np.array(bought_list) 
    recommended_list = np.array(recommended_list)[:k]
    flags = np.isin(bought_list, recommended_list) 
    return (flags.sum() > 0) * 1


def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(recommended_list)

# def precision_at_k(recommended_list, bought_list, k=5):
#     return precision(recommended_list[:k], bought_list)


def precision_at_k(recommended_list, bought_list, k=5):    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)    
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]    
    flags = np.isin(bought_list, recommended_list)           
    return flags.sum() / len(recommended_list)


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_recommended.sum()


def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(bought_list)

# def recall_at_k(recommended_list, bought_list, k=5):
#     return recall(recommended_list[:k], bought_list)


def recall_at_k(recommended_list, bought_list, k=5):    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])    
    flags = np.isin(bought_list, recommended_list)       
    return flags.sum() / len(bought_list)


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    prices_bought = np.array(prices_bought)
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_bought.sum()


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]
    if len(relevant_indexes) == 0:
        return 0
    amount_relevant = len(relevant_indexes)
    relevant_indexes = relevant_indexes[relevant_indexes <= k]
    sum_ = sum(
        [precision_at_k(recommended_list, bought_list, k=index_relevant + 1) for index_relevant in relevant_indexes])
    return sum_ / amount_relevant


def map_k(recommended_lists, bought_lists, k=5):    
    sum_ = sum([ap_k(recommended_lists[i], bought_lists[i], k=k) for i in range(len(bought_lists))])    
    return sum_/len(bought_lists)


def dcg_at_k(recommended_list, bought_list, k=5):    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])    
    flags = np.isin(recommended_list, bought_list).astype(int)    
    sum_ = sum([flags[i]/np.log2(i+1) if i+1 > 2 else flags[i]/(i+1) for i in range(k)])    
    return sum_/k


def ndcg_at_k(recommended_list, bought_list, k=5):    
    dcg = dcg_at_k(recommended_list, bought_list, k=k)
    ideal_dcg = sum([1/np.log2(i+1) if i+1 > 2 else 1/(i+1) for i in range(k)])/k    
    return dcg/ideal_dcg


def two_cycles(first_list, second_list):
    for i in first_list:
        for j in second_list:
            yield i, j
            
            
def reciprocal_rank(recommended_list, bought_list, k=5):    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]    
    rank = None    
    for i, bought in two_cycles(range(len(recommended_list)), bought_list):
        if recommended_list[i] == bought:
            rank = i+1
            break    
    if rank is None:
        return 0    
    return 1/rank


def mrr_k(recommended_lists, bought_lists, k=5):    
    sum_ = sum([reciprocal_rank(recommended_lists[i], bought_lists[i], k=k) for i in range(len(bought_lists))])    
    return sum_/len(bought_lists)
