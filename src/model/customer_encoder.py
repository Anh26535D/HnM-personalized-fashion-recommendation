import torch
from model.additive import AdditiveAttention


class CustomerEncoder(torch.nn.Module):
    def __init__(self, config):
        super(CustomerEncoder, self).__init__()
        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.num_filters)

    def forward(self, bought_articles):
        """
        Args:
            bought_articles: batch_size, num_purchased_articles_a_user, num_filters
        Returns:
            (shape) batch_size, num_filters
        """
        customer_vector = self.additive_attention(bought_articles)
        return customer_vector