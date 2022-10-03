import torch
import numpy as np
from mlp import MLP
from torch import nn
from torch_geometric.nn import knn_graph
from torch_geometric.nn import MessagePassing


def norm(u):
    """
    Computes the euclidean norm of a vector
    :param u: vector
    :return: euclidean norm of u
    """
    return np.sqrt(np.sum(u**2))

def compute_distance_matrix(locations):
    """
    Compute the distance matrix for all residue pairs.
    :param locations: numpy array of size (N_residues,3) of all C alpha positions
    :return: a symmetric numpy array of size (N_residues, N_residues) of pairwise distances
    """
    N_residues = len(locations)
    distance_matrix = np.zeros((N_residues, N_residues))
    for i, pos in enumerate(locations):
        distance_matrix[i,i] = np.inf
        for j in range(i+1, N_residues):
            distance_matrix[i, j] = norm(pos - locations[j,:])

    distance_matrix += distance_matrix.T
    return distance_matrix

class MessagePassingNetwork(MessagePassing):
    def __init__(self, message_mlp, update_mlp, aa_slice, non_aa_slice,  aa_size = 20, aa_embedding = 30):
        super().__init__(aggr="add", flow="source_to_target")
        self.aa_slice = aa_slice
        self.non_aa_slice = non_aa_slice
        self.aa_embedding_lin = nn.Linear(aa_size, aa_embedding, bias = False)
        self.message_mlp = message_mlp
        self.update_mlp = update_mlp

    def forward(self, x, edge_index, edge_attr, latent_variables):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, latent_variables=latent_variables)
        return out

    def message(self, x_i, x_j, edge_attr, latent_variables):
        #aa_embedding_i = self.aa_embedding_lin(x_i[:, self.aa_slice])
        #aa_embedding_j = self.aa_embedding_lin(x_j[:, self.aa_slice])
        #x_i = torch.cat((x_i[:, self.non_aa_slice], aa_embedding_i), dim = 1)
        #x_j = torch.cat((x_j[:, self.non_aa_slice], aa_embedding_j), dim=1)
        print(x_i[inte, :])
        print(x_j[inte, :])
        latent_variables = torch.broadcast_to(latent_variables, (6000, 5))
        x = torch.cat((x_i, x_j, edge_attr, latent_variables), dim = 1)
        return self.message_mlp.forward(x)

    def update(self, aggregated_i, latent_variables):
        latent_variables = torch.broadcast_to(latent_variables, (300, 5))
        x = torch.cat((aggregated_i, latent_variables), dim = 1)
        return self.update_mlp.forward(x)


NUM_NODES = 300
nodes_features = torch.normal(mean=torch.zeros((NUM_NODES, 3)), std=torch.ones((NUM_NODES, 3)))

edge_indexes = knn_graph(nodes_features, k=20, flow="source_to_target")
edge_attr = torch.normal(mean=torch.zeros((6000, 3)), std=torch.ones((6000, 3)))
print(edge_indexes)
inte = np.random.randint(0, 6000)
print(edge_indexes[:, 0])
neighb, node = edge_indexes[:, inte]

print(nodes_features[node, :])
print(nodes_features[neighb, :])
print("\n")

latent_variables = torch.normal(mean=torch.zeros((1, 5)), std=torch.ones((1, 5)))
message_mlp = MLP(14, 20, 100, num_hidden_layers=3)
update_mlp = MLP(25, 20, 100, num_hidden_layers=1)
mpn = MessagePassingNetwork(message_mlp, update_mlp, aa_slice=None, non_aa_slice=None)
res = mpn(nodes_features, edge_indexes, edge_attr, latent_variables)

#edge_index = knn_graph(nodes_features, 20)
#message_mlp = MLP()




