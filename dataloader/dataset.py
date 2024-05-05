import pickle
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from dataloader.read_dataset import RobotcarImageRead, RobotcarPCRead, RobotcarIMGPCRead


class PairwiseImageData(Dataset):
    def __init__(self, config):
        # load pair relation from the pickle file
        config = config.dataset
        pair_file = config.train_pickle
        image_folder = config.image_folder
        base_path = Path(config.base_path)
        pickle_path = Path(config.pickle_path)
        with open(pickle_path / Path(pair_file), 'rb') as file:
            self.pairs = pickle.load(file)

        # data reader
        self.reader = RobotcarImageRead(
            self.pairs['file_indices'],
            base_path,
            image_folder,
            config.augmentation,
        )
        self.posNum = len(self.pairs['pos_pairs'])
        self.negNum = len(self.pairs['neg_pairs'])
        self.random_num = torch.randint(200, 7000, (1,)).item()

    def __len__(self):
        return self.posNum

    def __getitem__(self, index):
        pos_indices = self.pairs['pos_pairs'][index]
        neg_indices = self.pairs['neg_pairs'][(index * self.random_num) % self.negNum]

        result = {
            'pos_pair': (
                self.reader.read_data(pos_indices[0]),
                self.reader.read_data(pos_indices[1]),
            ),
            'neg_pair': (
                self.reader.read_data(neg_indices[0]),
                self.reader.read_data(neg_indices[1]),
            ),
        }
        return result

    def shuffle_data(self, number):
        self.random_num = number


class TestImageData(Dataset):
    def __init__(self, config):
        # load test items matching relation from the pickle file
        config = config.dataset
        image_folder = config.image_folder
        pickle_file = Path(config.test_pickle)
        base_path = Path(config.base_path)
        pickle_path = Path(config.pickle_path)
        with open(pickle_path / pickle_file, 'rb') as f:
            self.items_pickle = pickle.load(f)

        file_indices = self.items_pickle['file_indices']
        self.num_of_each_run = [len(file_indices[i]) for i in range(len(file_indices))]

        self.combine_indices = []
        for i in range(len(file_indices)):
            self.combine_indices.extend(file_indices[i])

        # data reader
        self.reader = RobotcarImageRead(
            self.combine_indices,
            base_path,
            image_folder,
            augment=False,
        )

    def __len__(self):
        return sum(self.num_of_each_run)

    def __getitem__(self, index):
        return self.reader.read_data(index)

    def get_pos_items(self):
        return self.items_pickle['pos_items']

    def get_num_of_each_run(self):
        return self.num_of_each_run

    def get_all_test_file_names(self):
        return self.combine_indices

    def get_file_indices(self):
        return self.items_pickle['file_indices']


class PairwisePCData(Dataset):
    def __init__(self, config):
        # load pair relation from the pickle file
        config = config.dataset
        pair_file = config.train_pickle

        base_path = Path(config.base_path)
        pickle_path = Path(config.pickle_path)
        with open(pickle_path / Path(pair_file), 'rb') as file:
            self.pairs = pickle.load(file)

        # data reader
        self.reader = RobotcarPCRead(
            self.pairs['file_indices'],
            base_path,
            config.pc_folder,
            config.augmentation
        )
        self.posNum = len(self.pairs['pos_pairs'])
        self.negNum = len(self.pairs['neg_pairs'])
        self.random_num = torch.randint(200, 7000, (1,)).item()

    def __len__(self):
        return self.posNum

    def __getitem__(self, index):
        pos_indices = self.pairs['pos_pairs'][index]
        neg_indices = self.pairs['neg_pairs'][(index * self.random_num) % self.negNum]

        result = {
            'pos_pair': (
                self.reader.read_data(pos_indices[0]),
                self.reader.read_data(pos_indices[1]),
            ),
            'neg_pair': (
                self.reader.read_data(neg_indices[0]),
                self.reader.read_data(neg_indices[1]),
            ),
        }
        return result

    def shuffle_data(self, number):
        self.random_num = number


class TestPCData(Dataset):
    def __init__(self, config):
        # load test items matching relation from the pickle file
        config = config.dataset
        pickle_file = Path(config.test_pickle)
        base_path = Path(config.base_path)
        pickle_path = Path(config.pickle_path)
        with open(pickle_path / pickle_file, 'rb') as f:
            self.items_pickle = pickle.load(f)

        file_indices = self.items_pickle['file_indices']
        self.num_of_each_run = [len(file_indices[i]) for i in range(len(file_indices))]

        # combine each seperate run
        self.combine_indices = []
        for i in range(len(file_indices)):
            self.combine_indices.extend(file_indices[i])

        # data reader
        self.reader = RobotcarPCRead(
            self.combine_indices,
            base_path,
            config.pc_folder,
            augment=False
        )

    def __len__(self):
        return sum(self.num_of_each_run)

    def __getitem__(self, index):
        return self.reader.read_data(index)

    def get_pos_items(self):
        return self.items_pickle['pos_items']

    def get_num_of_each_run(self):
        return self.num_of_each_run

    def get_all_test_file_names(self):
        return self.combine_indices

    def get_file_indices(self):
        return self.items_pickle['file_indices']


class PairwiseIMGPCData(Dataset):
    def __init__(self, config):
        config = config.dataset
        pair_file = config.train_pickle
        base_path = Path(config.base_path)
        pickle_path = Path(config.pickle_path)
        with open(pickle_path / Path(pair_file), 'rb') as file:
            self.pairs = pickle.load(file)

        self.reader = RobotcarIMGPCRead(
            self.pairs['file_indices'],
            base_path,
            config.image_folder,
            config.pc_folder,
            config.augmentation,
        )
        neg_ratio = 1
        self.posNum = len(self.pairs['pos_pairs'])
        self.negNum = len(self.pairs['neg_pairs'])
        self.select_negNum = int(neg_ratio * self.posNum)

    def __len__(self):
        """Return the length of the dataset."""
        return self.posNum + self.select_negNum

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if index >= self.posNum:
            id = random.randint(0, self.negNum - 1)
            neg_indices = self.pairs['neg_pairs'][id]
            out = {
                "pc0": self.reader.read_pc_data(int(neg_indices[0])),
                "pc1": self.reader.read_pc_data(int(neg_indices[1])),
                "img0": self.reader.read_img_data(int(neg_indices[0])),
                "img1": self.reader.read_img_data(int(neg_indices[1])),
                'label': 0.}
            return out

        pos_indices = self.pairs['pos_pairs'][index]
        out = {
            "pc0": self.reader.read_pc_data(int(pos_indices[0])),
            "pc1": self.reader.read_pc_data(int(pos_indices[1])),
            "img0": self.reader.read_img_data(int(pos_indices[0])),
            "img1": self.reader.read_img_data(int(pos_indices[1])),
            'label': 1.}
        return out
