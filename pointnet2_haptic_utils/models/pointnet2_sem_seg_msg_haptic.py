import torch.nn as nn
import torch.nn.functional as F
from haptic.Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation


class get_shared_model(nn.Module):
    def __init__(self, use_batch_norm, num_classes, num_input_channel=7):
        super(get_shared_model, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], num_input_channel, [[16, 16, 32], [32, 32, 64]], use_batch_norm=use_batch_norm)
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]], use_batch_norm=use_batch_norm)
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]], use_batch_norm=use_batch_norm)
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]], use_batch_norm=use_batch_norm)
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256], use_batch_norm=use_batch_norm)
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256], use_batch_norm=use_batch_norm)
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128], use_batch_norm=use_batch_norm)
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128], use_batch_norm=use_batch_norm)
        self.conv1 = nn.Conv1d(128, 128, 1)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(128)
            self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        # for normal prediction
        self.conv_normal = nn.Conv1d(128, 3, 1)
        # for force prediction
        self.conv_force = nn.Conv1d(128, 1, 1)
        self.use_batch_norm = use_batch_norm

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        if self.use_batch_norm:
            x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        else:
            x = F.relu(self.conv1(l0_points))
        
        contact = self.conv2(x)
        normal = self.conv_normal(x)
        normal = F.normalize(normal, dim=1)
        force = self.conv_force(x)
        # this is not needed with BCElogit loss
        # x = F.log_softmax(x, dim=1)
        contact = contact.permute(0, 2, 1)
        normal = normal.permute(0, 2, 1)
        force = force.permute(0, 2, 1)
        return (contact, normal, force), l4_points

class get_model(nn.Module):
    def __init__(self, use_batch_norm, num_out_channel, num_in_channel=7, target='contact',
            radius_list=[[0.05, 0.1], [0.1, 0.2], [0.2, 0.4], [0.4, 0.8]], 
            npoint_list=[1024, 256, 64, 16],
            sample_point_1_list=[16, 16, 16, 16], 
            sample_point_2_list=[32, 32, 32, 32],
            layer=4,
            downsample=True,
            dropout=True,
            track_running_stats=True,
            mlp1_size=[16, 16, 32],  
            mlp2_size=[32, 32, 64],
            interpolation_mlp_size=[128, 128, 128]
            ):
        
        print("using layer: ", layer)
        super(get_model, self).__init__()
        self.layer = layer
        if self.layer == 4:
            self.sa1 = PointNetSetAbstractionMsg(npoint_list[0], radius_list[0], [sample_point_1_list[0], sample_point_2_list[0]], num_in_channel, [[16, 16, 32], [32, 32, 64]], use_batch_norm=use_batch_norm)
            self.sa2 = PointNetSetAbstractionMsg(npoint_list[1], radius_list[1], [sample_point_1_list[1], sample_point_2_list[1]], 32+64, [[64, 64, 128], [64, 96, 128]], use_batch_norm=use_batch_norm)
            self.sa3 = PointNetSetAbstractionMsg(npoint_list[2], radius_list[2], [sample_point_1_list[2], sample_point_2_list[2]], 128+128, [[128, 196, 256], [128, 196, 256]], use_batch_norm=use_batch_norm)
            self.sa4 = PointNetSetAbstractionMsg(npoint_list[3], radius_list[3], [sample_point_1_list[3], sample_point_2_list[3]], 256+256, [[256, 256, 512], [256, 384, 512]], use_batch_norm=use_batch_norm)
            self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256], use_batch_norm=use_batch_norm)
            self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256], use_batch_norm=use_batch_norm)
            self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128], use_batch_norm=use_batch_norm)
            self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128], use_batch_norm=use_batch_norm)
        elif self.layer == 3:
            self.sa1 = PointNetSetAbstractionMsg(npoint_list[0], radius_list[0], [sample_point_1_list[0], sample_point_2_list[0]], num_in_channel, [[16, 16, 32], [32, 32, 64]], use_batch_norm=use_batch_norm)
            self.sa2 = PointNetSetAbstractionMsg(npoint_list[1], radius_list[1], [sample_point_1_list[1], sample_point_2_list[1]], 32+64, [[64, 64, 128], [64, 96, 128]], use_batch_norm=use_batch_norm)
            self.sa3 = PointNetSetAbstractionMsg(npoint_list[2], radius_list[2], [sample_point_1_list[2], sample_point_2_list[2]], 128+128, [[128, 196, 256], [128, 196, 256]], use_batch_norm=use_batch_norm)
            self.fp3 = PointNetFeaturePropagation(128+128+256+256, [256, 256], use_batch_norm=use_batch_norm)
            self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128], use_batch_norm=use_batch_norm)
            self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128], use_batch_norm=use_batch_norm)
        elif self.layer == 2:
            self.sa1 = PointNetSetAbstractionMsg(npoint_list[0], radius_list[0], [sample_point_1_list[0], sample_point_2_list[0]], num_in_channel, [[16, 16, 32], [32, 32, 64]], use_batch_norm=use_batch_norm)
            self.sa2 = PointNetSetAbstractionMsg(npoint_list[1], radius_list[1], [sample_point_1_list[1], sample_point_2_list[1]], 32+64, [[64, 64, 128], [64, 96, 128]], use_batch_norm=use_batch_norm)
            self.fp2 = PointNetFeaturePropagation(32+64+128+128, [256, 128], use_batch_norm=use_batch_norm)
            self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128], use_batch_norm=use_batch_norm)
        elif self.layer == 1:
            self.sa1 = PointNetSetAbstractionMsg(npoint_list[0], radius_list[0], [sample_point_1_list[0], sample_point_2_list[0]], num_in_channel, [mlp1_size, mlp2_size], use_batch_norm=use_batch_norm,
                downsample=downsample, track_running_stats=track_running_stats)
            self.fp1 = PointNetFeaturePropagation(mlp1_size[-1] + mlp2_size[-1], interpolation_mlp_size, use_batch_norm=use_batch_norm, track_running_stats=track_running_stats)

        self.drop_out = dropout
        self.conv1 = nn.Conv1d(128, 128, 1)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(128, track_running_stats=track_running_stats)
            if self.drop_out:
                self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_out_channel, 1)
        self.use_batch_norm = use_batch_norm
        self.target = target

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        if self.layer == 4:
            l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
            l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

            l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
            l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
            l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
            l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        elif self.layer == 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

            l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
            l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
            l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        elif self.layer == 2:
            l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

            l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
            l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        elif self.layer == 1:
            l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
            l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        if self.use_batch_norm:
            if self.drop_out:
                x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
            else:
                x = F.relu(self.bn1(self.conv1(l0_points)))
        else:
            x = F.relu(self.conv1(l0_points))
        
        x = self.conv2(x)
        # this is not needed with BCElogit loss
        # x = F.log_softmax(x, dim=1)
        if self.target == 'normal':
            x = F.normalize(x, dim=1)
        x = x.permute(0, 2, 1)
        # return x, l4_points
        return x, None


class get_loss_original(nn.Module):
    def __init__(self):
        super(get_loss_original, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = self.loss(pred, target)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))