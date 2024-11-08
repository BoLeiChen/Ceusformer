import torch
import torch.nn as nn
import torch.nn.functional as F

from torchkeras import summary
from model_v1.backbone.vgg16 import VGG16
from torch.utils.tensorboard import SummaryWriter


class Double_CBR(nn.Module):
    """
    Double_CBR 是 Conv BN Relu 两次堆叠
    """

    def __init__(self, in_channel, out_channel, is_pooling=False):
        super(Double_CBR, self).__init__()
        self.is_pooling = is_pooling
        self.CBR2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6()
        )
        if self.is_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        if self.is_pooling:
            x = self.pool(x)
        x = self.CBR2(x)
        return x


class Unet_Skip_Up(nn.Module):
    """
    Unet_Skip_Up 是 Unet 跳跃链接模块
    """

    def __init__(self, in_channel, out_channel):
        super(Unet_Skip_Up, self).__init__()
        self.CBR2 = Double_CBR(in_channel, out_channel, is_pooling=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, feat_encoder, feat_up):
        feat_up = self.upsample(feat_up)
        output_feat = torch.cat([feat_encoder, feat_up], dim=1)
        output_feat = self.CBR2(output_feat)
        return output_feat


class Unet_Encoder(nn.Module):
    def __init__(self, in_channel=3):
        super(Unet_Encoder, self).__init__()
        self.CBR_512 = Double_CBR(in_channel, 32, is_pooling=False)
        self.CBR_256 = Double_CBR(32, 64, is_pooling=True)
        self.CBR_128 = Double_CBR(64, 128, is_pooling=True)
        self.CBR_64 = Double_CBR(128, 256, is_pooling=True)
        self.CBR_32 = Double_CBR(256, 256, is_pooling=True)

    def forward(self, x):
        feat_512 = self.CBR_512(x)
        feat_256 = self.CBR_256(feat_512)
        feat_128 = self.CBR_128(feat_256)
        feat_64 = self.CBR_64(feat_128)
        feat_32 = self.CBR_32(feat_64)

        return feat_512, feat_256, feat_128, feat_64, feat_32


class Att_block(nn.Module):
    def __init__(self, channel_g, channel_x, channel_mid):
        super(Att_block, self).__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(channel_g, channel_mid, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel_mid)
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(channel_x, channel_mid, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel_mid)
        )
        self.weight = nn.Sequential(
            nn.Conv2d(channel_mid, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, g, x):
        g1 = self.upsample(g)
        g1 = self.Wg(g1)
        x1 = self.Wx(x)
        psi = F.relu6(g1 + x1)
        weight = self.weight(psi)

        return weight * x


class MLP(torch.nn.Module):

    def __init__(self, dims):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(dims[0], dims[1])
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(dims[1], dims[2])  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(dims[2], dims[3])

    def forward(self, x):
        x = torch.flatten(x.squeeze(1), start_dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


class Attention_Unet_Vgg(nn.Module):
    """
    Attention_Unet_Vgg 是在跳跃链接处加入了 注意力门机制, 使用下层特征指导跳跃链接的上层特征, 具体原理参见论文。

    parameter: num_class: 默认二分类
    parameter: in_channels: 默认彩图
    parameter: pretrain: 是否进行迁移学习

    下采样： Maxpooling
    上采样： UpsampleBilinear
    """

    def __init__(self, num_class=2, in_channels=3):
        super(Attention_Unet_Vgg, self).__init__()
        print(self.__doc__)
        self.encoder = Unet_Encoder(3)
        self.skip_up_64 = Unet_Skip_Up(512, 128)
        self.skip_up_128 = Unet_Skip_Up(256, 64)
        self.skip_up_256 = Unet_Skip_Up(128, 32)
        self.skip_up_512 = Unet_Skip_Up(64, 32)
        self.att_block_64 = Att_block(256, 256, 128)
        self.att_block_128 = Att_block(128, 128, 64)
        self.att_block_256 = Att_block(64, 64, 32)
        self.cls_conv = nn.Conv2d(32, num_class, kernel_size=(3, 3), padding=1)

        self.label_pred_conv_1 = nn.Conv2d(num_class, 1, kernel_size=(3, 3), padding=1)
        self.label_pred_conv_2 = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1)
        self.label_pred_conv_3 = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1)
        self.label_pred_conv_4 = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1)
        self.label_head = MLP([32*32, 512, 64, 1])
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        feat_512, feat_256, feat_128, feat_64, feat_32 = self.encoder(x)
        feat_64_w = self.att_block_64(feat_32, feat_64)
        up_64 = self.skip_up_64(feat_64_w, feat_32)

        feat_128_w = self.att_block_128(up_64, feat_128)
        up_128 = self.skip_up_128(feat_128_w, up_64)

        feat_256_w = self.att_block_256(up_128, feat_256)
        up_256 = self.skip_up_256(feat_256_w, up_128)

        up_512 = self.skip_up_512(feat_512, up_256)

        results_seg = self.cls_conv(up_512)

        label_pred = self.pool(self.label_pred_conv_1(results_seg))
        label_pred = self.pool(self.label_pred_conv_2(label_pred))
        label_pred = self.pool(self.label_pred_conv_3(label_pred))
        label_pred = self.pool(self.label_pred_conv_4(label_pred))
        label_pred = self.label_head(label_pred)

        results = {"seg_us": results_seg, "label": label_pred}
        return results


if __name__ == "__main__":
    attention_unet_vgg = Attention_Unet_Vgg(num_class=2, in_channels=3)
    summary(attention_unet_vgg, (3, 512, 512))

    # dummy_input = torch.randn(1, 3, 512, 512)
    # outputs = base_unet_vgg_official(dummy_input)
    # print(outputs.shape)

    # writer = SummaryWriter("arch_plot/" + base_unet_vgg_official._get_name())
    # writer.add_graph(base_unet_vgg_official, torch.randn((1, 3, 512, 512)))
    # writer.close()
