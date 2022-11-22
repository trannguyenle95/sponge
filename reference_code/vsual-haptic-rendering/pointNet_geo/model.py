import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate
import torch.nn as nn

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import torch_geometric.transforms as T
from torch_geometric.nn import PointConv, fps, radius, global_max_pool

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip

class Net(torch.nn.Module):
    def __init__(self, args, num_classes, 
            num_layer=3, 
            feature_num=4, 
            sa_radius=[0.2, 0.4],
            sa_ratio=[0.2, 0.25],  
            sa_mlp_list=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
            fp_mlp_list=[[256, 256], [256, 128], [128, 128, 128]],
            linear_mlp_list=[128, 128],
            fp_k=[1, 3, 3],
            residual=False
        ):
        super(Net, self).__init__()

        # Input channels account for both `pos` and node features.
        self.residual = residual
        self.num_layer = num_layer
        self.args = args

        self.sa_module_list = nn.ModuleList()
        self.sa_module_list.append(SAModule(sa_ratio[0], sa_radius[0], MLP([feature_num + 3, *sa_mlp_list[0]], batch_norm=args.use_batch_norm)))
        for l_idx in range(1, self.num_layer - 1):
            self.sa_module_list.append(SAModule(sa_ratio[l_idx], sa_radius[l_idx], MLP([sa_mlp_list[l_idx - 1][-1] + 3, *sa_mlp_list[l_idx]], batch_norm=args.use_batch_norm)))
        
        self.sa_module_list.append(GlobalSAModule(MLP([sa_mlp_list[self.num_layer - 2][-1] + 3, *sa_mlp_list[self.num_layer - 1]], batch_norm=args.use_batch_norm)))
            
        self.fp_module_list = nn.ModuleList()
        self.fp_module_list.append(FPModule(fp_k[0], MLP([sa_mlp_list[self.num_layer-1][-1] + sa_mlp_list[self.num_layer-2][-1], *fp_mlp_list[0]], batch_norm=args.use_batch_norm)))
        for l_idx in range(self.num_layer-2, 0, -1):
            f_idx = self.num_layer - 1 - l_idx
            self.fp_module_list.append(FPModule(fp_k[f_idx], MLP([fp_mlp_list[f_idx-1][-1] + sa_mlp_list[l_idx-1][-1], *fp_mlp_list[f_idx]], batch_norm=args.use_batch_norm)))
        self.fp_module_list.append(FPModule(fp_k[self.num_layer-1], MLP([sa_mlp_list[0][-1] + feature_num, *fp_mlp_list[self.num_layer-1]], batch_norm=args.use_batch_norm)))

        self.lin_layers = nn.ModuleList()
        in_size = fp_mlp_list[-1][-1]
        for size in linear_mlp_list:
            self.lin_layers.append(torch.nn.Linear(in_size, size))
            in_size = size
        self.out_layer = torch.nn.Linear(in_size, num_classes)

    def forward(self, x, pos, batch):
        sa_out = (x, pos, batch)
        sa_outs = [sa_out]
        for i in range(self.num_layer):
            sa_out = self.sa_module_list[i](*sa_out)
            sa_outs.append(sa_out)

        fp_out = self.fp_module_list[0](*sa_outs[-1], *sa_outs[-2])
        for i in range(1, self.num_layer):
            fp_out = self.fp_module_list[i](*fp_out, *sa_outs[-(i+2)])

        x, _, _ = fp_out

        for layer in self.lin_layers:
            if not self.residual:
                x = F.relu(layer(x))
            else:
                x = x + F.relu(layer(x))
            if self.args.use_batch_norm:
                x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.out_layer(x)
        return x

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv((x, x[idx]), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    if batch_norm:
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
            for i in range(1, len(channels))
        ])
    else:
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i]), ReLU())
            for i in range(1, len(channels))
        ])