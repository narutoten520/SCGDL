import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True 
cudnn.benchmark = True 
import torch.nn.functional as F
from torch_geometric.nn import ResGatedGraphConv

class ResGatedGraphmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(ResGatedGraphmodel, self).__init__()

        
        [in_channels, num_hidden, out_channels] = hidden_dims #3000 128 128 
        self.conv1 = ResGatedGraphConv(in_channels, num_hidden) 

        self.conv2 = ResGatedGraphConv(num_hidden, num_hidden)

        self.conv3 = ResGatedGraphConv(num_hidden, num_hidden)

        self.conv4 = ResGatedGraphConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 
        
    def forward(self, data):
        Activation_f = F.elu    
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)
        return h4 