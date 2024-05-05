import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

base_path = '/xxx/datasets/oxford/'  # your oxford robotCar dataset location

filename = "pointcloud_locations_20m_10overlap.csv"

all_folders = sorted(os.listdir(base_path))
folders = []

# All runs are used for training (both full and partial)
index_list = range(len(all_folders))
print("Number of runs: " + str(len(index_list)))
for index in index_list:
    folders.append(all_folders[index])
print(folders)

#####For training and test data split#####
x_width = 150
y_width = 150
p1 = [5735712.768124, 620084.402381]
p2 = [5735611.299219, 620540.270327]
p3 = [5735237.358209, 620543.094379]
p4 = [5734749.303802, 619932.693364]
p = [p1, p2, p3, p4]


def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if (point[0] - x_width < northing and northing < point[0] + x_width and
            point[1] - y_width < easting and easting < point[1] + y_width):
            in_test_set = True
            break
    return in_test_set


def construct_training_dict(df_centroids, save_filename):
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=10)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=50)
    pos_pairs = []
    neg_pairs = []
    file_indices = []
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        positives = np.setdiff1d(ind_nn[i], [i]).tolist()
        negatives = np.setdiff1d(df_centroids.index.values.tolist(), ind_r[i]).tolist()
        for m in positives:
            if i < m:
                pair_pos = [i, m]
                pos_pairs.append(pair_pos)
        for n in negatives:
            if i < n:
                pair_neg = [i, n]
                neg_pairs.append(pair_neg)
        file_indices.append(query)

    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)

    queries = {"file_indices": file_indices,
               "pos_pairs": pos_pairs,
               "neg_pairs": neg_pairs}

    with open(save_filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", save_filename)


def construct_test_dict(df_centroids, save_filename):
    file_indices = []
    pos_items = {}
    for i in range(len(df_centroids)):
        seq_i_file = df_centroids[i]['file'].tolist()
        file_indices.append(seq_i_file)

        for j in range(i + 1, len(df_centroids)):
            conbine_ij_centroids = df_centroids[i].append(df_centroids[j])
            tree = KDTree(conbine_ij_centroids[['northing', 'easting']])
            ind_nn = tree.query_radius(conbine_ij_centroids[['northing', 'easting']], r=20)
            pos_ij = []
            for k in range(len(ind_nn)):
                pos_ij.append(ind_nn[k])

            pos_items[(i, j)] = pos_ij

    queries = {"file_indices": file_indices,
               "pos_items": pos_items}

    with open(save_filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", save_filename)


# Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
df_tests = []  # split accord to folder
for folder in folders:
    df_locations = pd.read_csv(os.path.join(base_path, folder, filename), sep=',')
    df_locations['timestamp'] = folder + '/' + df_locations['timestamp'].astype(str)
    df_locations = df_locations.rename(columns={'timestamp': 'file'})

    df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])
    for index, row in df_locations.iterrows():
        if check_in_test_set(row['northing'], row['easting'], p, x_width, y_width):
            df_test = df_test.append(row, ignore_index=True)
        else:
            df_train = df_train.append(row, ignore_index=True)
    df_tests.append(df_test)

print("Number of training submaps: " + str(len(df_train['file'])))
print("Number of non-disjoint test submaps: " + str(sum([len(df_tests[i]) for i in range(len(df_tests))])))

# FOR TRAINING
construct_training_dict(df_train, "training_queries_baseline.pickle")

# FOR TEST
construct_test_dict(df_tests, "test_queries_baseline.pickle")

