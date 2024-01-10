import argparse
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class GraphConv(nn.Module):
    """
    graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(1))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h = torch.matmul(input, self.W) # W是转换矩阵，w0用于对节点本身变换
       
        # 邻接矩阵中，值为1的位置，值不为1的位置被置-9e15
        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SemGraphConv_(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0]) # W是转换矩阵，w0用于对节点本身变换
        h1 = torch.matmul(input, self.W[1]) # w1用于对其他节点变换

        # 邻接矩阵中，值为1的位置，值不为1的位置被置-9e15
        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


def symmetric_normalize_adjacency(adjacency_matrix):
    # Compute degree matrix
    degrees = np.sum(adjacency_matrix, axis=1)
    degree_matrix = np.diag(np.power(degrees, -0.5))
    
    # Symmetric normalization
    normalized_adjacency = np.dot(np.dot(degree_matrix, adjacency_matrix), degree_matrix)
    return normalized_adjacency

def symmetric_normalize_adjacency_matrix(adjacency_matrix):
    # Calculate the degree matrix
    degree_matrix = torch.sum(adjacency_matrix, dim=1)
    degree_matrix_pow = torch.pow(degree_matrix, -0.5)
    
    # Avoiding division by zero
    degree_matrix_pow[degree_matrix_pow == float('inf')] = 0
    
    # Symmetric normalization
    normalized_adjacency = torch.matmul(torch.matmul(torch.diag(degree_matrix_pow), adjacency_matrix), torch.diag(degree_matrix_pow))
    return normalized_adjacency

class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = (self.adj > 0)
        # self.adj = symmetric_normalize_adjacency_matrix(self.adj)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(1))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h = torch.matmul(input, self.W) # W是转换矩阵，w0用于对节点本身变换
       
        # 邻接矩阵中，值为1的位置，值不为1的位置被置-9e15
        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        output = torch.matmul(adj, h)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class TSemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, adj, node_dim, in_features, out_features,  bias=True):
        super(TSemGraphConv, self).__init__()
        self._input_dim = in_features
        self.out_features = out_features
        self.node_dim = node_dim
        
        self.W = nn.Parameter(torch.zeros(size=(2, in_features+node_dim, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, node_vector, hidden_state):
        batch_size, num_nodes, dim_node = node_vector.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        # inputs = inputs.reshape((batch_size, num_nodes, 1))
        
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._input_dim)
        )
        # [x, h] (batch_size, num_nodes, _input_dim + dim_node)
        input = torch.cat((node_vector, hidden_state), dim=2)

        h0 = torch.matmul(input, self.W[0]) # W是转换矩阵，w0用于对节点本身变换
        h1 = torch.matmul(input, self.W[1]) # w1用于对其他节点变换

        # 邻接矩阵中，值为1的位置，值不为1的位置被置-9e15
        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            output += self.bias.view(1, 1, -1)
        
        return output.view(batch_size, -1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class TSemGCNCell(nn.Module):
    def __init__(self, adj, node_dim:int, input_dim: int, hidden_dim: int):
        super(TSemGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TSemGraphConv(
            self.adj, node_dim, self._hidden_dim, self._hidden_dim * 2
        )
        self.graph_conv2 = TSemGraphConv(
            self.adj, node_dim, self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):
        # inputs - bs, num_nodes, dim_node
        # hidden_state - 
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}

class TSemGCN(nn.Module):
    def __init__(self, adj, node_dim: int, hidden_dim: int, **kwargs):
        super(TSemGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TSemGCNCell(self.adj, node_dim, self._hidden_dim, self._hidden_dim)
        self.outlayer = SemGraphConv(self._hidden_dim, 2, adj)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, _ = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        outputs = []
        feats = []
        for i in range(seq_len):
            # input - bs, num_nodes, dim_node
            output, hidden_state = self.tgcn_cell(inputs[:, i, :, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            feats.append(output)
            output = self.outlayer(output)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)    
        return {'pred': outputs, 'feat': feats}

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
    
