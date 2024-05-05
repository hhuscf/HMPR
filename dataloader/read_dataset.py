import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torch


class RobotcarImageRead:
    def __init__(
            self,
            file_indices, basePath, image_folder, augment=True):
        self.file_indices = file_indices
        self.basePath = Path(basePath)
        self.IMG_FOLDER = Path(image_folder)
        self.augment = augment

        # transform operation
        if augment:
            self.image_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            )

        self.image_normalizer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def data_augment(self, image):
        image = self.image_jitter(image)
        return image

    def read_data(self, ind: int):
        # parse file names
        folder, timestamp = self.file_indices[ind].split('/')
        folder = self.basePath / Path(folder)
        image_path = folder / self.IMG_FOLDER / Path(timestamp + '.png')

        # read data
        image_PIL = Image.open(image_path)
        # crop the hood of car about 40 pixels in height to make 320x240 to 320x200
        box = (0, 0, 320, 200)
        image_PIL = image_PIL.crop(box)

        # data augmentation
        if self.augment:
            image_PIL = self.data_augment(image_PIL)
        # prepare output
        image_tensor = self.image_normalizer(image_PIL)

        return image_tensor


class RobotcarPCRead:
    def __init__(
            self,
            file_indices, basePath, pcFolder, augment=True):
        self.file_indices = file_indices
        self.basePath = Path(basePath)
        self.augment = augment
        self.PC_FOLDER = Path(pcFolder)
        # transform operation
        if augment:
            self.pc_sigma = 0.02
            self.pc_clip = 0.05

    def data_augment(self, pc):
        jitter_cloud = np.clip(
            self.pc_sigma * np.random.randn(*pc.shape).astype(np.float32),
            -1.0 * self.pc_clip,
            self.pc_clip,
        )  # jitter point cloud
        return jitter_cloud + pc

    def read_data(self, ind: int):
        # parse file names
        folder, timestamp = self.file_indices[ind].split('/')
        folder = self.basePath / Path(folder)
        pc_path = folder / self.PC_FOLDER / Path(timestamp + '.bin')

        pc_ndarray = np.fromfile(pc_path, dtype=np.float64).astype(np.float32)
        pc_ndarray.resize(pc_ndarray.shape[0] // 3, 3)

        # data augmentation
        if self.augment:
            pc_ndarray = self.data_augment(pc_ndarray)
        pc_tensor = torch.from_numpy(pc_ndarray)
        return pc_tensor


class RobotcarIMGPCRead:
    def __init__(self, file_indices, basePath, image_folder, pc_folder, augment: bool = True):
        self.file_indices = file_indices
        self.basePath = Path(basePath)
        self.augment = augment
        self.PC_FOLDER = Path(pc_folder)
        self.IMG_FOLDER = Path(image_folder)
        self.pc_sigma = 0.02
        self.pc_clip = 0.05
        # transform operation
        if augment:
            self.image_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            )

        self.image_normalizer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def data_augment(self, pc):
        jitter_cloud = np.clip(
            self.pc_sigma * np.random.randn(*pc.shape).astype(np.float32),
            -1.0 * self.pc_clip,
            self.pc_clip,
        )  # jitter point cloud
        return jitter_cloud + pc

    def data_augment_img(self, image):
        image = self.image_jitter(image)
        return image

    def read_pc_data(self, ind: int):
        # parse file names
        folder, timestamp = self.file_indices[ind].split('/')
        folder = self.basePath / Path(folder)
        pc_path = folder / self.PC_FOLDER / Path(timestamp + '.bin')
        pc_ndarray = np.fromfile(pc_path, dtype=np.float64).astype(np.float32)
        pc_ndarray.resize(pc_ndarray.shape[0] // 3, 3)
        # data augmentation
        if self.augment:
            pc_ndarray = self.data_augment(pc_ndarray)
        pc_tensor = torch.from_numpy(pc_ndarray)
        return pc_tensor

    def read_img_data(self, ind: int):
        # parse file names
        folder, timestamp = self.file_indices[ind].split('/')
        folder = self.basePath / Path(folder)
        image_path = folder / self.IMG_FOLDER / Path(timestamp + '.png')
        # read data
        image_PIL = Image.open(image_path)
        # crop the hood of car about 40 pixel in height to make 320x240 to 320x200
        box = (0, 0, 320, 200)
        image_PIL = image_PIL.crop(box)
        # data augmentation
        if self.augment:
            image_PIL = self.data_augment_img(image_PIL)
        # prepare output
        image_tensor = self.image_normalizer(image_PIL)
        return image_tensor

