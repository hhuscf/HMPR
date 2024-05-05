import numpy as np
from sklearn.neighbors import KDTree


def recall_atN(features: np.ndarray, pos_items: list, N: int, Lp: int):
    success_num = 0
    total_num = len(pos_items)
    tree = KDTree(features, p=Lp)

    # recall@N, +1 means find itself and delete itself below with np.setdiff1d
    ind_n1 = tree.query(features, k=N + 1, return_distance=False)

    for i in range(len(ind_n1)):
        if not pos_items[i].any():  # skip empty
            total_num -= 1
            continue

        retrieved_items = set(np.setdiff1d(ind_n1[i], [i]))
        if retrieved_items & set(pos_items[i]):
            success_num += 1

    recall_1 = success_num / total_num

    # recall@1%
    success_num1 = 0
    total_num1 = len(pos_items)
    N1 = max(int(round(len(pos_items) / 100.0)), 1)

    ind_n1 = tree.query(features, k=N1 + 1, return_distance=False)
    for i in range(len(ind_n1)):
        if not pos_items[i].any():  # skip empty
            total_num1 -= 1
            continue

        retrieved_items = set(np.setdiff1d(ind_n1[i], [i]))
        if retrieved_items & set(pos_items[i]):
            success_num1 += 1

    recall_1percent = success_num1 / total_num1

    return recall_1, recall_1percent


