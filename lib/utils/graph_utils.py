import torch
import numpy as np
import scipy.sparse as sp

def adj_matrix_from_num_points(adj_dir, num_points):

    connection = np.loadtxt(adj_dir)
       
    return adj_mx_from_list(connection, num_points)

def adj_mx_from_list(connection, num_points, is_weight=False):
    if is_weight:
        print("=> using weighted graph")
        weights = connection[:, 2]
    else:
        weights = None
    edges = connection[:, :2]
    return adj_mx_from_weighted_edges(num_points, edges, weights, sparse=False)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def adj_mx_from_weighted_edges(num_pts, edges, weights=None, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    if weights is None:
        weights = np.ones(edges.shape[0])
    data, i, j = weights, edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx

def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


def main():
    matrix = adj_matrix_from_num_points("A2C", 41, False)
    W, H = matrix.shape
    print(W,H)
    with open('matrix.txt', 'w') as f:
        for w in range(W):
            for h in range(H):
                f.write(str(float(matrix[w][h]))+' ')
            f.write('\n')

if __name__ == '__main__':
    main()
