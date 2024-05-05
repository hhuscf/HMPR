import os
import pickle
import sys
import time
import argparse
from sklearn.neighbors import KDTree
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader.dataset import TestImageData
from models.coarsenet import ImageCoarseNet
from utils.utils import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="config directory")
parser.add_argument("--ckpt_file", type=str, required=True, help="checkpoint location")
args = parser.parse_args()

config = load_config(args.config_path)
test_loader = DataLoader(
    TestImageData(config),
    batch_size=config.test_batch_size,
    shuffle=False,
    num_workers=config.workers,
)


def searchKDTree(features: np.ndarray, N: int, Lp: int):
    tree = KDTree(features, p=Lp)
    # +1 means find itself and delete itself below with np.setdiff1d
    ind_n1 = tree.query(features, k=N + 1, return_distance=False)
    indexs = [np.setdiff1d(ind_n1[i], [i], assume_unique=True) for i in range(len(ind_n1))]

    return indexs


def evaluate_coarse_topk(features, topk, save_filename):
    # get some variables
    pos_items = test_loader.dataset.get_pos_items()
    num_of_each_run = test_loader.dataset.get_num_of_each_run()
    sum_num_of_each_run = [
        sum(num_of_each_run[:i]) for i in range(len(num_of_each_run))
    ]
    run_num = len(num_of_each_run)

    # compute evaluation metric
    coarse_index = {}

    pairs = ([i, j] for i in range(run_num) for j in range(i + 1, run_num))
    for i, j in pairs:
        st1 = sum_num_of_each_run[i]
        st2 = sum_num_of_each_run[j]
        end1 = st1 + num_of_each_run[i]
        end2 = st2 + num_of_each_run[j]

        feature_of_two_run = np.vstack((features[st1:end1], features[st2:end2]))

        indexs = searchKDTree(feature_of_two_run, N=topk, Lp=config.loss.Lp)

        coarse_index[(i, j)] = indexs

    # record test results
    query_indexs = {"file_index": test_loader.dataset.get_file_indices(),
                    "coarse_index": coarse_index,
                    "pos_index": pos_items}
    with open(save_filename, 'wb') as handle:
        pickle.dump(query_indexs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved topk = ", topk, " candidates index for fine eval")


def calculate_feature(net):
    features = []
    net.eval()
    total_time = 0.
    print('Calculate coarse features...')
    with torch.no_grad():
        time1 = time.time()
        for img in test_loader:
            img = img.to(device)
            batch_feature = net(img)
            batch_feature = batch_feature.detach().cpu().numpy()
            features.append(batch_feature)
        total_time += (time.time() - time1)
    print("Feature generation time per frame", total_time / len(test_loader.dataset))
    features = np.vstack(features)
    return features


# generate Top-K candidates for subsequent Fine Net
if __name__ == '__main__':
    coarse_net = ImageCoarseNet(config).to(device)
    resume_filename = args.ckpt_file
    assert os.path.exists(resume_filename), 'Cannot open network weights: {}'.format(resume_filename)
    print("Resuming From ", resume_filename)
    checkpoint = torch.load(resume_filename, map_location=device)
    state_dict = checkpoint['state_dict']
    coarse_net.load_state_dict(state_dict, strict=False)

    # calculate coarse feature
    features = calculate_feature(coarse_net)

    # save topK candidates
    k = 15
    save_pickle_name = './eval_queries_fine_index.pickle'
    evaluate_coarse_topk(features, k, save_pickle_name)
