import torch
from torch.nn import functional as F

from model.article_encoder import ArticleEncoder
from model.customer_encoder import CustomerEncoder


class DotProductClickPredictor(torch.nn.Module):
    def __init__(self):
        super(DotProductClickPredictor, self).__init__()

    def forward(self, candidate_articles, customer_vector):
        """
        Args:
            candidate_articles: batch_size, candidate_size, num_filters
            customer_vector: batch_size, num_filters
        Returns:
            (shape): batch_size
        """
        # batch_size, candidate_size
        probability = torch.bmm(candidate_articles,
                                customer_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        return probability


class NAML(torch.nn.Module):
    """
    NAML network.
    Input 1 + K candidate articles and a list of bought articles, produce the click probability.
    """
    def __init__(self, config, pretrained_word_embedding=None):
        super(NAML, self).__init__()
        self.config = config
        self.article_encoder = ArticleEncoder(config, pretrained_word_embedding)
        self.customer_encoder = CustomerEncoder(config)
        self.click_predictor = DotProductClickPredictor()

    def forward(self, candidates, bought_articles):
        """
        Args:
            candidates:
                [
                    {
                        category_cols: batch_size,
                        text_cols: batch_size * num_words_[text_cols]},
                    } * (1 + K)
                ]
            bought_articles:
                [
                    {
                        category_cols: batch_size,
                        text_cols: batch_size * num_words_[text_cols]},
                    } * num_purchased_articles_a_user
                ]
        Returns:
            click_probability: batch_size
        """
        # batch_size, 1 + K, num_filters
        _candidate_articles = torch.stack(
            [self.article_encoder(x) for x in candidates], dim=1)
        # batch_size, num_purchased_articles_a_user, num_filters
        _bought_articles = torch.stack(
            [self.article_encoder(x) for x in bought_articles], dim=1)
        
        # batch_size, num_filters
        customer_vector = self.customer_encoder(_bought_articles)
        # batch_size, 1 + K
        click_probability = self.click_predictor(_candidate_articles,
                                                 customer_vector)
        return click_probability

    def get_article_vector(self, articles):
        """
        Args:
            articles:
                {
                    category_cols: batch_size,
                    text_cols: batch_size * num_words_[text_cols]},
                }
        Returns:
            (shape) batch_size, num_filters
        """
        # batch_size, num_filters
        return self.article_encoder(articles)

    def get_customer_vector(self, bought_article_vector):
        """
        Args:
            bought_article_vector: batch_size, num_purchased_articles_a_user, num_filters
        Returns:
            (shape) batch_size, num_filters
        """
        # batch_size, num_filters
        return self.customer_encoder(bought_article_vector)

    def get_prediction(self, article_vector, customer_vector):
        """
        Args:
            article_vector: candidate_size, num_filters
            customer_vector: num_filters
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        return self.click_predictor(
            article_vector.unsqueeze(dim=0),
            customer_vector.unsqueeze(dim=0)).squeeze(dim=0)