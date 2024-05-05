import torch
import torch.nn as nn
import torchvision.models as models


class ResnetFeatureExtractor(nn.Module):
    def __init__(self, pool_method='gem', return_loc_fea=False):
        super(ResnetFeatureExtractor, self).__init__()
        self.pool_method = pool_method
        self.return_loc_fea = return_loc_fea
        model = models.resnet18(pretrained=True)
        # get rid of Last 2 blocks
        self.resnet_fe = nn.ModuleList(list(model.children())[:-2])

        if self.pool_method == 'gem':
            self.pool = GeM()
        elif self.pool_method == 'spoc':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.pool_method == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise NotImplementedError("Unknown pooling method: {}".format(self.pool_method))

    def forward(self, x):
        x = self.resnet_fe[0](x)
        x = self.resnet_fe[1](x)
        x = self.resnet_fe[2](x)
        x = self.resnet_fe[3](x)
        x = self.resnet_fe[4](x)
        x = self.resnet_fe[5](x)
        x = self.resnet_fe[6](x)

        if self.return_loc_fea:
            return x
        x = self.resnet_fe[7](x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
