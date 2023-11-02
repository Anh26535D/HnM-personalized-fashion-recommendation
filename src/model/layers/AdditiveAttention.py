import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(torch.nn.Module):
    def __init__(self,
                 candidate_vector_dim,
                 query_vector_dim,
                 ):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, candidate_vector: torch.Tensor) -> torch.Tensor:
        """

        Arguments
        ----------
            candidate_vector: torch.Tensor
                Tensor with shape of (batch_size, candidate_size, candidate_vector_dim).

        Returns
        -------
            torch.Tensor
                Tensor with shape (batch_size, candidate_vector_dim).
        """
        # temp has shape of [batch_size, candidate_size, query_vector_dim]
        temp = torch.tanh(self.linear(candidate_vector))

        # candidate_weights has shape of [batch_size, candidate_size]
        candidate_weights = torch.matmul(temp, self.attention_query_vector)
        candidate_weights = F.softmax(candidate_weights, dim=1)

        # target has shape of [batch_size, candidate_vector_dim]
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target
