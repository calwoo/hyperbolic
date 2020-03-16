from manifolds.euclidean import Euclidean
from manifolds.hyperbolic import Hyperbolic
from data import parse_zachary, generate_random_walks

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepWalk(nn.Module):
    """
    Implementation of a deep-walk model. This is effectively nothing but a
        random walk -> "sentences" -> skip-gram model
    to produce latent embeddings of graphs. We will implement this model with
    negative contrastic estimation (negative sampling) in order to improve
    training rates.
    """

    def __init__(self, n_vertices, embedding_dim=5, geometry="euclidean"):
        self.n_vertices = n_vertices
        self.embedding_dim = embedding_dim

        if geometry == "euclidean":
            self.geom = Euclidean(dim=embedding_dim)
        elif geometry == "hyperbolic":
            self.geom = Hyperbolic(dim=embedding_dim)

        super(DeepWalk, self).__init__()
        self.input_embed = nn.Embedding(n_vertices, embedding_dim)
        self.output_embed = nn.Embedding(n_vertices, embedding_dim)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, target, context, negatives) -> torch.Tensor:
        """
        Forward pass of skipgram model. Uses negative sampling
        to train, where the logit scores are computed using the
        underlying geometry's Riemannian inner product.

        :param target: [batch_size, 1] Tensor of target vertices.
        :param context: [batch_size, 1] Tensor of context vertices.
        :param negatives: [batch_size, n_negs] Tensor of negative samples
            for negative contrastive estimation.
        :return torch.Tensor: Estimated NLL loss for the skipgram model.
        """

        # assert sizes
        target, context = target.view((-1, 1)), context.view((-1, 1))
        assert len(target.size()) == 2 and len(context.size()) == 2

        u = self.input_embed(target)
        v = self.output_embed(context)
        pos_nll = self.log_sigmoid(self.geom.dot(u, v)).squeeze()

        v_negs = self.output_embed(negatives) # [batch_size, n_negs, embedding_dim]
        neg_nll = 0
        for t in range(v_neg.size(1)):
            v_neg = v_negs[:, t, :].squeeze(1)
            assert len(v_neg.size()) == 2

            



if __name__ == "__main__":
    edges = parse_zachary("zachary.txt")
    walks = generate_random_walks(edges, walk_length=10)
    n_vertices = len(edges.keys())

    model = DeepWalk(n_vertices=n_vertices)
