import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from utils.data_utils import *


class FuseSRmodel(nn.Module):
    def __init__(self, setting):
        super(FuseSRmodel, self).__init__()
        self.scale_factor = setting["upsample_factor"]

        
        feature_channels = 16
        primary_channels = 2*feature_channels
        current_channels = 32
        backbone_channels = current_channels
        inv_channels = 3 * self.scale_factor * self.scale_factor

        self.encoder = Encoder(feature_channels)

        self.CurrentExpend = nn.Sequential(
            nn.Conv2d(feature_channels, current_channels, 3, padding=1),
            nn.ReLU()
        ) 

        self.backbone = nn.Sequential(
            nn.Conv2d(backbone_channels + inv_channels, backbone_channels, 3, padding=1),
            nn.ReLU(),
            utils.InvertedResidual(backbone_channels, backbone_channels, 1, 1),
            utils.InvertedResidual(backbone_channels, backbone_channels, 1, 1),
            utils.InvertedResidual(backbone_channels, backbone_channels, 1, 1),
            utils.InvertedResidual(backbone_channels, backbone_channels, 1, 1),
            utils.InvertedResidual(backbone_channels, backbone_channels, 1, 1),
            utils.InvertedResidual(backbone_channels, backbone_channels, 1, 1),
        )

        self.Upsampling = utils.PixelShuffleUpsampling(backbone_channels + primary_channels, 3, self.scale_factor)

        self.refine_conv = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, data : dict):
        current = torch.cat((data[irradiance_name], data[depth_name], data[normal_name]), dim = 1)

        shallow_features, current_features = self.encoder(current)
        current_features = self.CurrentExpend(current_features)

        BRDF = data[precomputed_BRDF_name + HR_postfix_name]

        backbone_output = self.backbone(
                    torch.cat((current_features, F.pixel_unshuffle(BRDF, self.scale_factor)), dim=1))

        output = self.refine_conv(
            self.Upsampling(torch.cat((backbone_output, shallow_features), dim=1))
            ) * BRDF

        return output

class Encoder(nn.Module):
    def __init__(self, out_channels):
        super(Encoder, self).__init__()

        channel1 = out_channels * 2
        channel2 = out_channels
        channel3 = (out_channels * 3) //4

        self.primary_encoder = nn.Sequential(
            nn.Conv2d(7, channel1, 3, padding=1),
            nn.ReLU()
        )

        self.main_encoder = nn.Sequential(
            utils.DWSConvolution(channel1, channel2, 3, 1, 1),
            nn.BatchNorm2d(channel2),
            nn.ReLU(),
            utils.DWSConvolution(channel2, channel3, 3, 1, 1),
            nn.BatchNorm2d(channel3),
            nn.ReLU(),
            utils.DWSConvolution(channel3, channel3, 3, 1, 1),
            nn.BatchNorm2d(channel3),
            nn.ReLU(),
            utils.DWSConvolution(channel3, channel2, 3, 1, 1),
            nn.BatchNorm2d(channel2),
            nn.ReLU()
        )

    def forward(self, inputs):
        primary_feature = self.primary_encoder(inputs)
        feature = self.main_encoder(primary_feature)
        return primary_feature, feature