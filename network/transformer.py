import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        h = F.relu(self.W_in(h_V))
        h = self.W_out(h)
        return h


class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_latent, n_neighbors, num_heads=4):
        """

        :param num_hidden: dimension of the hidden node representation
        :param num_in: dimension of the input to the key track (usually num_hidden + dim_edges_features)
        :param num_heads: number of heads
        """
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_latent = num_latent
        self.n_neighbors = n_neighbors

        # Self-attention layers: {queries, keys, values, output, latent, edges bias}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)
        ##Add biases or not ??
        self.W_z = nn.Linear(num_latent, int(n_neighbors*num_heads), bias=False)
        self.W_b = nn.Linear(num_in - 33, num_heads, bias=False)

        return

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """
        Numerically stable masked softmax
        :param attend_logits: quantities used to compute the softmax, [n_batch, n_nodes, n_heads, n_neighbors]
        :param mask_attend: mask that gives where these quantities should be -inf, [N_batch, N_nodes, N_heads, n_neighbors]
        :param dim: on which dimension to compute the logit.
        :return: softmax on the masked quantities, ruling out non neighbors.
        """
        negative_inf = np.finfo(np.float32).min
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf))
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend * attend
        return attend

    def forward(self, z, h_V, h_E, mask_attend=None):
        """ Self-attention, graph-structured O(Nk)
        Args:
            z:              latent variables        [N_batch, N_latent]
            h_V:            Node features           [N_batch, N_nodes, N_hidden]
            h_E:            Neighbor features       [N_batch, N_nodes, K, N_hidden]
            mask_attend:    Mask for attention      [N_batch, N_nodes, K]
        Returns:
            h_V:            Node update
        """

        # Queries, Keys, Values
        n_batch, n_nodes, n_neighbors = h_E.shape[:3]
        n_heads = self.num_heads

        d = int(self.num_hidden / n_heads)
        #Maps to the query, key and value space. We start with one vector and map it to different
        ##Query, key and values. This can be done in one transformation and then reshaping if we assume
        ##every head has the same dimension.
        Q = self.W_Q(h_V).view([n_batch, n_nodes, 1, n_heads, 1, d])
        K = self.W_K(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d, 1])
        V = self.W_V(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d])
        bias_latent = self.W_z(z).view([n_batch, 1, n_neighbors, n_heads]).transpose(-2, -1)
        bias_edges = self.W_b(h_E[:n_batch, :n_nodes, :n_neighbors, 33:]).view([n_batch, n_nodes, n_neighbors, n_heads]).transpose(-2, -1)

        # Attention with scaled inner product. The attend_logits was oragnized s.t for each neighbor, we have
        #all the heads values. But obviously, we want that for each head, we have all the neighbors, so that
        #we can actually multiply with the value vector. [n_batch, n_nodes, n_heads, n_neighbors]
        attend_logits = torch.matmul(Q, K).view([n_batch, n_nodes, n_neighbors, n_heads]).transpose(-2 ,-1)
        attend_logits = attend_logits / np.sqrt(d) + bias_latent + bias_edges

        if mask_attend is not None:
            # Masked softmax
            # Mask first becomes of size [N_batch, N_nodes, 1, K] and then [N_batch, N_nodes, N_head, K]
            # This makes sense: we repeat the same masking on the K neighbors across all the heads.
            mask = mask_attend.unsqueeze(2).expand(-1 ,-1 ,n_heads ,-1)
            attend = self._masked_softmax(attend_logits, mask)
        else:
            #attend is of size: [n_batch, n_nodes, n_heads, n_neighbors]
            attend = F.softmax(attend_logits, -1)

        # Attentive reduction
        #Matmul between tensors of size [n_batch, n_nodes, n_heads, 1, n_neighbors]
        #and [n_batch, n_nodes, n_heads, n_neighbors, d]. Result of size
        #[n_batch, n_nodes, n_heads, d]
        h_V_update = torch.matmul(attend.unsqueeze(-2), V.transpose(2 ,3))
        h_V_update = h_V_update.view([n_batch, n_nodes, self.num_hidden])
        h_V_update = self.W_O(h_V_update)
        return h_V_update

    def step(self, t, h_V, h_E, E_idx, mask_attend=None):
        """ Self-attention for a specific time step t
        Args:
            h_V:            Node features           [N_batch, N_nodes, N_hidden]
            h_E:            Neighbor features       [N_batch, N_nodes, K, N_in]
            E_idx:          Neighbor indices        [N_batch, N_nodes, K]
            mask_attend:    Mask for attention      [N_batch, N_nodes, K]
        Returns:
            h_V_t:            Node update
        """
        # Dimensions
        n_batch, n_nodes, n_neighbors = h_E.shape[:3]
        n_heads = self.num_heads
        d = self.num_hidden / n_heads

        # Per time-step tensors
        h_V_t = h_V[: ,t ,:]
        h_E_t = h_E[: ,t ,: ,:]
        E_idx_t = E_idx[: ,t ,:]

        # Single time-step
        h_V_neighbors_t = gather_nodes_t(h_V, E_idx_t)
        E_t = torch.cat([h_E_t, h_V_neighbors_t], -1)

        # Queries, Keys, Values
        Q = self.W_Q(h_V_t).view([n_batch, 1, n_heads, 1, d])
        K = self.W_K(E_t).view([n_batch, n_neighbors, n_heads, d, 1])
        V = self.W_V(E_t).view([n_batch, n_neighbors, n_heads, d])

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K).view([n_batch, n_neighbors, n_heads]).transpose(-2 ,-1)
        attend_logits = attend_logits / np.sqrt(d)

        if mask_attend is not None:
            # Masked softmax
            # [N_batch, K] -=> [N_batch, N_heads, K]
            mask_t = mask_attend[: ,t ,:].unsqueeze(1).expand(-1 ,n_heads ,-1)
            attend = self._masked_softmax(attend_logits, mask_t)
        else:
            attend = F.softmax(attend_logits / np.sqrt(d), -1)

        # Attentive reduction
        h_V_t_update = torch.matmul(attend.unsqueeze(-2), V.transpose(1 ,2))
        return h_V_t_update


class TransformerLayer(nn.Module):
    def __init__(self, num_hidden, num_in, num_latent, n_neighbors, num_heads=4, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])

        self.attention = NeighborAttention(num_hidden, num_in,  num_latent, n_neighbors, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        # Self-attention
        dh = self.attention(h_V, h_E, mask_attend)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

    def step(self, t, h_V, h_E, mask_V=None, mask_attend=None):
        """ Sequential computation of step t of a transformer layer """
        # Self-attention
        h_V_t = h_V[:,t,:]
        dh_t = self.attention.step(t, h_V, h_E, mask_attend)
        h_V_t = self.norm[0](h_V_t + self.dropout(dh_t))

        # Position-wise feedforward
        dh_t = self.dense(h_V_t)
        h_V_t = self.norm[1](h_V_t + self.dropout(dh_t))

        if mask_V is not None:
            mask_V_t = mask_V[:,t].unsqueeze(-1)
            h_V_t = mask_V_t * h_V_t
        return h_V_t