import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from utils.data_utils import *

class FuseSRmodel(nn.Module):
    def __init__(self, setting):
        super(FuseSRmodel, self).__init__()
        self.scale_factor = setting["upsample_factor"]

        self.encoder = Encoder()

        self.CurrentExpend = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        self.attention = MotionVectorAttention()

        backbone_channels = 128

        # inv_channels = 3 * self.scale_factor * self.scale_factor
        inv_channels = (2 * 3 + 1) * self.scale_factor * self.scale_factor

        self.HRCompact = nn.Sequential(
            nn.Conv2d(inv_channels, 64, 3, padding=1),
            nn.ReLU()
        )

        # backbone_input_channels = backbone_channels + inv_channels
        backbone_input_channels = backbone_channels + 64
        self.backbone = nn.Sequential(
            nn.Conv2d(backbone_input_channels, backbone_channels, 3, padding=1),
            nn.ReLU(),
            ResBlock(backbone_channels),
            ResBlock(backbone_channels),
            ResBlock(backbone_channels),
            ResBlock(backbone_channels),
            ResBlock(backbone_channels),
            ResBlock(backbone_channels)
        )

        self.Upsampling = utils.PixelShuffleUpsampling(backbone_channels + 64, 3, self.scale_factor)

        self.refine_conv = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, data : dict):
        current = torch.cat((data[input_name], data[depth_name], data[normal_name]), dim = 1)

        shallow_features, current_features = self.encoder(current)
        current_features = self.CurrentExpend(current_features)

        historical_1 = torch.cat((
            data["history_1_" + input_name],
            data["history_1_" + depth_name], 
            data["history_1_" + normal_name]), dim = 1)

        motion_vector_1 = data[motion_name]

        historical_1_features = utils.backward_warping(self.encoder(historical_1)[1], motion_vector_1)

        historical_1_features = historical_1_features * self.attention(torch.cat((motion_vector_1, data["history_1_" + depth_name], data[depth_name]), dim=1))
        
        historical_2 = torch.cat((
            data["history_2_" + input_name],
            data["history_2_" + depth_name], 
            data["history_2_" + normal_name]), dim = 1)

        motion_vector_2 = data["history_1_" + motion_name]
        
        motion_vector_2 += utils.backward_warping(motion_vector_1, motion_vector_2)

        historical_2_features = utils.backward_warping(self.encoder(historical_2)[1], motion_vector_2)

        historical_2_features = historical_2_features * self.attention(torch.cat((motion_vector_2, data["history_2_" + depth_name], data[depth_name]), dim=1))

        historical_features = torch.cat((historical_1_features, historical_2_features), dim=1)

        # BRDF = data[precomputed_BRDF_name + HR_postfix_name]
        depthHR = data[depth_name + HR_postfix_name]
        normalHR = data[normal_name + HR_postfix_name]
        albedoHR = data[albedo_name + HR_postfix_name]
        # HR_features = self.HRCompact(F.pixel_unshuffle(torch.cat((BRDF, depthHR, normalHR, albedoHR), dim=1), self.scale_factor))
        HR_features = self.HRCompact(F.pixel_unshuffle(torch.cat((albedoHR, depthHR, normalHR), dim=1), self.scale_factor))

        # backbone_output = self.backbone(
        #             torch.cat((current_features, historical_features, F.pixel_unshuffle(BRDF, self.scale_factor)), dim=1))
        backbone_output = self.backbone(torch.cat((current_features, historical_features, HR_features), dim=1))

        output = self.refine_conv(
            self.Upsampling(torch.cat((backbone_output, shallow_features), dim=1))
            ) # No demodulation

        return output

class MotionVectorAttention(nn.Module):
    def __init__(self):
        super(MotionVectorAttention, self).__init__()
        channel = 4
        self.W_k = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.ReLU()
        )

        self.W_q = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.ReLU()
        )

        self.Va_q = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, inputs):
        return self.Va_q(torch.tanh(torch.sum(self.W_k(inputs) + self.W_q(inputs), dim=1).unsqueeze(1)))

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.shallow_encoder = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1),
            nn.ReLU()
        )

        self.main_encoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, inputs):
        shallow_feature = self.shallow_encoder(inputs)
        feature = self.main_encoder(shallow_feature)
        return shallow_feature, feature

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, inputs):
        return self.block(inputs) + inputs