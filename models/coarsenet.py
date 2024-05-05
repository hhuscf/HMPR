import torch
import torch.nn as nn
from models.resnet import ResnetFeatureExtractor
from models.pointnet2 import PointNet2


class ImageCoarseNet(nn.Module):
    """Resnet18 coarse image PR net"""
    def __init__(self, config=None, return_loc_fea=False):
        super(ImageCoarseNet, self).__init__()
        self.config = config
        self.return_loc_fea = return_loc_fea
        self.feature_extract = ResnetFeatureExtractor(pool_method=self.config.model.pool_method,
                                                      return_loc_fea=self.return_loc_fea)

    def forward(self, img):
        return self.feature_extract(img)


class GeM1D(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM1D, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool1d(x.clamp(min=self.eps).pow(self.p), x.size(-1)).pow(1./self.p)


class PointCloudCoarseNet(nn.Module):
    """ Pointnet++ PointCloud PR net"""
    def __init__(self, config=None, return_loc_fea=False):
        super(PointCloudCoarseNet, self).__init__()
        self.config = config
        self.return_loc_fea = return_loc_fea
        self.feature_extract = PointNet2(config=self.config)
        self.pool = GeM1D()

    def forward(self, pc):
        fea = self.feature_extract(pc)
        if self.return_loc_fea:
            return fea
        fea = self.pool(fea)
        fea = torch.flatten(fea, 1)
        return fea



