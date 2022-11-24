# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.utils.data
# import torch.nn.functional as F
# from models.pointnet import PointNetEncoder, feature_transform_reguliarzer


# class get_model(nn.Module):
#     def __init__(self, num_class=1, normal_channel=True):
#         super(get_model, self).__init__()
#         if normal_channel:
#             channel = 8
#         else:
#             channel = 5
#         self.k = num_class
#         self.feat_o = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)  # feature trans True
#         # self.feat_h = PointNetEncoder(global_feat=False, feature_transform=False, channel=channel)  # feature trans True
#         self.conv1 = torch.nn.Conv1d(1088, 512, 1)
#         self.conv2 = torch.nn.Conv1d(512, 256, 1)
#         self.conv3 = torch.nn.Conv1d(256, 128, 1)
#         self.conv4 = torch.nn.Conv1d(128, self.k, 1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.convfuse = torch.nn.Conv1d(3778, 3000, 1)
#         self.bnfuse = nn.BatchNorm1d(3000)

#     def forward(self, x):
#         '''
#         :param x: obj pc [B, D, N]
#         :return: regressed cmap
#         '''
#         batchsize = x.size()[0]
#         n_pts = x.size()[2]
#         # for obj
#         x, trans, trans_feat = self.feat_o(x)  # x: [B, 1088, N] global+point feature of object
#         # # for hand
#         # hand, trans2, trans_feat2 = self.feat_h(hand)  # hand: [B, 1088, 778] global+point feature of hand
#         # fuse feature of object and hand
#         # x = torch.cat((x, hand), dim=2).permute(0,2,1).contiguous()  # [B, N+778, 1088]
#         x = F.relu(self.bnfuse(self.convfuse(x)))  # [B, N, 1088]
#         x = x.permute(0,2,1).contiguous()  # [B, 1088, N]
#         # inference cmap
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, N]
#         x = self.conv4(x)  # [B, 1, N]
#         x = x.transpose(2,1).contiguous()
#         x = torch.sigmoid(x)
#         x = x.view(batchsize, n_pts)  # n_pts  [B, N]
#         return x

# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()

#     def forward(self, pred, target, trans_feat):
#         total_loss = F.nll_loss(pred, target)

#         return total_loss


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from models.pointnet import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self, num_class,normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
                channel = 8
        else:
            channel = 5
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=channel)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        # x = torch.sigmoid(x)
        x = x.view(batchsize, n_pts)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        # target = target.view(target.shape[0],target[1],1)
        loss = nn.BCEWithLogitsLoss()(pred, target)
        # print("Loss: ", loss)
        return loss


if __name__ == '__main__':
    model = get_model(13)
    xyz = torch.rand(12, 3, 2048)
    (model(xyz))