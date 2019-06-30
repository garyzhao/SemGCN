from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemCHGraphConv(nn.Module):
    """
    Semantic channel-wise graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemCHGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj.unsqueeze(0).repeat(out_features, 1, 1)
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(out_features, len(self.m[0].nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(1))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0]).unsqueeze(1).transpose(1, 3)  # B * C * J * 1
        h1 = torch.matmul(input, self.W[1]).unsqueeze(1).transpose(1, 3)  # B * C * J * 1

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)  # C * J * J
        adj[self.m] = self.e.view(-1)
        adj = F.softmax(adj, dim=2)

        E = torch.eye(adj.size(1), dtype=torch.float).to(input.device)
        E = E.unsqueeze(0).repeat(self.out_features, 1, 1)  # C * J * J
        output = torch.matmul(adj * E, h0) + torch.matmul(adj * (1 - E), h1)
        output = output.transpose(1, 3).squeeze(1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
