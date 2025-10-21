import torch
import torch.nn as nn
import numpy as np

from efficientnet_pytorch import EfficientNet
from model.convolutions import UpsamplingConcat, DeepLabHead
from torch.hub import load as torch_hub_load

class CamEncoder(nn.Module):
    def __init__(self, cfg, D):
        super().__init__()
        self.D = D
        self.C = cfg.bev_encoder_in_channel
        self.use_depth_distribution = cfg.use_depth_distribution
        self.downsample = cfg.bev_down_sample
        self.version = cfg.backbone.split('-')[1]

        self.backbone = EfficientNet.from_pretrained(cfg.backbone)
        self.delete_unused_layers()
        if self.version == 'b4':
            self.reduction_channel = [0, 24, 32, 56, 160, 448]
        elif self.version == 'b0':
            self.reduction_channel = [0, 16, 24, 40, 112, 320]
        else:
            raise NotImplementedError
        self.upsampling_out_channel = [0, 48, 64, 128, 512]

        index = np.log2(self.downsample).astype(int)

        if self.use_depth_distribution:
            self.depth_layer_1 = DeepLabHead(self.reduction_channel[index + 1],
                                             self.reduction_channel[index + 1],
                                             hidden_channel=64)
            self.depth_layer_2 = UpsamplingConcat(self.reduction_channel[index + 1] + self.reduction_channel[index],
                                                  self.D)

        self.feature_layer_1 = DeepLabHead(self.reduction_channel[index + 1],
                                           self.reduction_channel[index + 1],
                                           hidden_channel=64)
        self.feature_layer_2 = UpsamplingConcat(self.reduction_channel[index + 1] + self.reduction_channel[index],
                                                self.C)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features_depth(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        index = np.log2(self.downsample).astype(int)
        input_1 = endpoints['reduction_{}'.format(index + 1)]
        input_2 = endpoints['reduction_{}'.format(index)]

        feature = self.feature_layer_1(input_1)
        feature = self.feature_layer_2(feature, input_2)

        if self.use_depth_distribution:
            depth = self.depth_layer_1(input_1)
            depth = self.depth_layer_2(depth, input_2)
        else:
            depth = None

        return feature, depth

    def get_backbone_feature(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break
        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        index = np.log2(self.downsample).astype(int)
        input_1 = endpoints['reduction_{}'.format(index + 1)]
        input_2 = endpoints['reduction_{}'.format(index)]

        return input_1      

    def forward(self, x):
        feature, depth = self.get_features_depth(x)  # get feature vector

        # if self.use_depth_distribution:
        #     depth_prob = depth.softmax(dim=1)
        #     feature = depth_prob.unsqueeze(1) * feature.unsqueeze(2)  # outer product depth and features
        # else:
        #     feature = feature.unsqueeze(2).repeat(1, 1, self.D, 1, 1)

        return feature, depth

class DinoCamEncoder(nn.Module):
    def __init__(self, cfg, D):
        super().__init__()
        self.D = D
        self.C = cfg.bev_encoder_in_channel
        self.use_depth_distribution = cfg.use_depth_distribution
        self.downsample = cfg.bev_down_sample
        self.version = cfg.backbone  # should be 'vitl14_reg' etc.

        # Load DINOv2 from torch.hub
        self.backbone = torch_hub_load(
            'facebookresearch/dinov2',
            self.version,
            pretrained=True
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        # self.backbone.eval()

        # DINOv2 ViT-L/14 outputs features of shape (B, C, H, W), typically C = 1024
        if "vitl" in self.version:
            dinov2_out_channels = 1024
        elif "vitb" in self.version:
            dinov2_out_channels = 768
        else:
            raise NotImplementedError
            
        index = int(np.log2(self.downsample))

        # For DINOv2, just treat input_1 as the full feature map
        self.reduction_channel = [0] * 6
        self.reduction_channel[index + 1] = dinov2_out_channels
        self.reduction_channel[index] = dinov2_out_channels  # assume skip-connection of same or half channels

        if self.use_depth_distribution:
            self.depth_layer_1 = DeepLabHead(
                self.reduction_channel[index + 1],
                self.reduction_channel[index + 1],
                hidden_channel=64
            )
            self.depth_layer_2 = UpsamplingConcat(
                self.reduction_channel[index + 1] + self.reduction_channel[index],
                self.D,
                scale_factor=1
            )

        self.feature_layer_1 = DeepLabHead(
            self.reduction_channel[index + 1],
            self.reduction_channel[index + 1],
            hidden_channel=64
        )
        self.feature_layer_2 = UpsamplingConcat(
            self.reduction_channel[index + 1] + self.reduction_channel[index],
            self.C,
            scale_factor=1
        )

    def delete_unused_layers(self):
        # Not used in DINOv2
        pass

    def get_features_depth(self, x):
        # DINOv2 expects normalized images in [0, 1] with ImageNet stats

        # features = self.backbone.forward_features(x)
        # feat = features["x_prenorm"]  # shape: (B, C, H, W)
        feat = self.backbone.get_intermediate_layers(x, n=1, reshape=False)[0] # x.shape: [6,3,294,294], feat.shape: [6,441,768]

        # feat = feat[:, :-5, :]              # remove CLS and REG tokens → (B, 441, 1024)
        B, N, C = feat.shape
        H = W = int(N ** 0.5)
        feat = feat.permute(0, 2, 1).reshape(B, C, H, W)  # (B, 768, 21, 21)

        input_1 = feat
        input_2 = feat  # can use another layer if intermediate available

        feature = self.feature_layer_1(input_1) # feature.shape: [6, 768, 21, 21]
        feature = self.feature_layer_2(feature, input_2) # feature.shape: [6, 64, 21, 21]

        if self.use_depth_distribution:
            depth = self.depth_layer_1(input_1) # depth.shape: [6, 768, 21, 21]
            depth = self.depth_layer_2(depth, input_2) # depth.shape: [6, 48, 21, 21]
        else:
            depth = None

        return feature, depth

    def get_backbone_feature(self, x):
        # DINOv2 expects normalized images in [0, 1] with ImageNet stats

        # features = self.backbone.forward_features(x)
        # feat = features["x_prenorm"]  # shape: (B, C, H, W)
        feat = self.backbone.get_intermediate_layers(x, n=1, reshape=False)[0] # x.shape: [6,3,294,294], feat.shape: [6,441,768]
        B, N, C = feat.shape
        H = W = int(N ** 0.5)
        feat = feat.permute(0, 2, 1).reshape(B, C, H, W)  # (B, 768, 21, 21)

        return feat     


    def forward(self, x):
        feature, depth = self.get_features_depth(x)
        return feature, depth
