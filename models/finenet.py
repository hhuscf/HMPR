import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import MultiHeadAttention, M_PosAttention1d, S_ChnAttention1d
from models.coarsenet import PointCloudCoarseNet, ImageCoarseNet


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.dropout = dropout
        self.norm_1 = Norm(self.d_model)
        self.norm_2 = Norm(self.d_model)
        self.norm_3 = Norm(self.d_model)

        self.cross_attn0 = MultiHeadAttention(self.heads, self.d_model, dropout=self.dropout)
        self.cross_attn1 = MultiHeadAttention(self.heads, self.d_model, dropout=self.dropout)

        self.ff1 = FeedForward(self.d_model, dropout=self.dropout)
        self.ff2 = FeedForward(self.d_model, dropout=self.dropout)

    def forward(self, fa, fb):
        fa = fa.permute(0, 2, 1)
        fb = fb.permute(0, 2, 1)
        fa = F.normalize(fa, dim=-1)
        fb = F.normalize(fb, dim=-1)
        fa = self.norm_1(fa)
        fb = self.norm_1(fb)

        fa_hat = fa + self.cross_attn0(fa, fb, fb, mask=None)
        fa_hat = self.norm_2(fa_hat)
        fa_hat = fa_hat + self.ff1(fa_hat)

        fb_hat = fb + self.cross_attn1(fb, fa, fa, mask=None)
        fb_hat = self.norm_3(fb_hat)
        fb_hat = fb_hat + self.ff2(fb_hat)

        fa_hat = fa_hat.permute(0, 2, 1)
        fb_hat = fb_hat.permute(0, 2, 1)

        return fa_hat, fb_hat


class similarityFineNet(nn.Module):
    def __init__(self, config=None):
        super(similarityFineNet, self).__init__()
        self.config = config
        self.img_weight_path = self.config.pretrained_image_ckpt_path
        self.pc_weight_path = self.config.pretrained_pc_ckpt_path
        assert os.path.exists(self.img_weight_path), 'Cannot open network weights: {}'.format(self.img_weight_path)
        assert os.path.exists(self.pc_weight_path), 'Cannot open network weights: {}'.format(self.pc_weight_path)
        self.image_feature_extract = ImageCoarseNet(config=self.config, return_loc_fea=True)
        self.pc_feature_extract = PointCloudCoarseNet(config=self.config, return_loc_fea=True)
        # load weight and freeze weight
        checkpoint = torch.load(self.img_weight_path)
        state_dict = checkpoint['state_dict']
        self.image_feature_extract.load_state_dict(state_dict, strict=False)

        checkpoint = torch.load(self.pc_weight_path)
        state_dict = checkpoint['state_dict']
        self.pc_feature_extract.load_state_dict(state_dict, strict=False)

        for param in self.image_feature_extract.parameters():
            param.requires_grad = False
        for param in self.pc_feature_extract.parameters():
            param.requires_grad = False

        self.atten0 = AttentionBlock(heads=4, d_model=256, dropout=0.)
        self.pool0 = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Sequential(nn.Linear(in_features=512, out_features=256), nn.LeakyReLU(
            negative_slope=0.1), nn.Linear(in_features=256, out_features=1))

        self.atten1 = AttentionBlock(heads=4, d_model=256, dropout=0.)
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.v_attention_pos1 = M_PosAttention1d(
            256, kernel_size=3, dk=4, dv=64, Nh=1, stride=2, singleHead=True
        )
        self.v_attention_chn1 = S_ChnAttention1d(in_chn=256, out_chn=64)
        self.v_atten_fusion = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.l_attention_pos1 = M_PosAttention1d(
            256, kernel_size=3, dk=4, dv=64, Nh=1, stride=2, singleHead=True
        )
        self.l_attention_chn1 = S_ChnAttention1d(in_chn=256, out_chn=64)
        self.l_atten_fusion = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.vl_atten_fusion = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

    def forward(self, pc0, pc1, img0, img1):
        pc0 = self.pc_feature_extract(pc0)
        pc1 = self.pc_feature_extract(pc1)
        img0 = self.image_feature_extract(img0)
        img1 = self.image_feature_extract(img1)

        pc0, pc1 = self.atten0(pc0, pc1)
        pc_diff_fea = torch.abs(pc0 - pc1)
        pc_diff = self.pool0(pc_diff_fea).view(pc0.shape[0], -1)

        bs, channel, _, _ = img0.shape
        img0 = img0.view(bs, channel, -1)
        img1 = img1.view(bs, channel, -1)
        img0, img1 = self.atten1(img0, img1)
        img_diff_fea = torch.abs(img0 - img1)
        img_diff = self.pool1(img_diff_fea).view(img0.shape[0], -1)

        # add weight attention
        img_atten_pos1 = self.v_attention_pos1(img_diff_fea)
        img_atten_chn1 = self.v_attention_chn1(img_diff_fea)
        pc_atten_pos1 = self.l_attention_pos1(pc_diff_fea)
        pc_atten_chn1 = self.l_attention_chn1(pc_diff_fea)

        final_sz = img_atten_pos1.shape[2:]
        img_atten_chn1 = F.interpolate(img_atten_chn1, size=final_sz,
                                       mode='nearest')
        img_atten = self.v_atten_fusion(
            torch.cat(
                (
                    img_atten_pos1,
                    img_atten_chn1,
                ),
                dim=1,
            )
        ).squeeze(-1)

        final_sz = pc_atten_pos1.shape[2:]
        pc_atten_chn1 = F.interpolate(pc_atten_chn1, size=final_sz, mode='nearest')
        pc_atten = self.l_atten_fusion(
            torch.cat(
                (
                    pc_atten_pos1,
                    pc_atten_chn1,
                ),
                dim=1,
            )
        ).squeeze(-1)

        pc_img_atten = torch.cat((pc_atten, img_atten), dim=1)
        pc_img_atten = self.vl_atten_fusion(pc_img_atten).unsqueeze(2)

        pc_diff = pc_diff.unsqueeze(1)
        img_diff = img_diff.unsqueeze(1)
        pc_img_diff = torch.cat((pc_diff, img_diff), dim=1)

        out = torch.reshape(torch.mul(pc_img_atten, pc_img_diff), (pc_img_diff.shape[0], -1))
        out = self.linear(out).view(-1)
        return torch.sigmoid(out), torch.norm(pc_img_diff, dim=1)

