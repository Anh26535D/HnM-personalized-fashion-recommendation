import torch
import torch.nn as nn
import torch.nn.functional as F
from model.additive import AdditiveAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TextEncoder(torch.nn.Module):
    def __init__(self, word_embedding, word_embedding_dim, num_filters,
                 window_size, query_vector_dim, dropout_probability):
        super(TextEncoder, self).__init__()
        self.word_embedding = word_embedding
        self.dropout_probability = dropout_probability
        self.CNN = nn.Conv2d(1,
                             num_filters, (window_size, word_embedding_dim),
                             padding=(int((window_size - 1) / 2), 0))
        self.additive_attention = AdditiveAttention(query_vector_dim,
                                                    num_filters)

    def forward(self, text):
        # batch_size, num_words_text, word_embedding_dim
        text_vector = F.dropout(self.word_embedding(text),
                                p=self.dropout_probability,
                                training=self.training)
        # batch_size, num_filters, num_words_title
        convoluted_text_vector = self.CNN(
            text_vector.unsqueeze(dim=1)).squeeze(dim=3)
        # batch_size, num_filters, num_words_title
        activated_text_vector = F.dropout(F.relu(convoluted_text_vector),
                                          p=self.dropout_probability,
                                          training=self.training)

        # batch_size, num_filters
        text_vector = self.additive_attention(
            activated_text_vector.transpose(1, 2))
        return text_vector


class CategoryEncoder(torch.nn.Module):
    def __init__(self, embedding, linear_input_dim, linear_output_dim):
        super(CategoryEncoder, self).__init__()
        self.embedding = embedding
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)

    def forward(self, element):
        return F.relu(self.linear(self.embedding(element)))


class ArticleEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(ArticleEncoder, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            word_embedding = nn.Embedding(config.num_words,
                                          config.word_embedding_dim,
                                          padding_idx=0)
        else:
            word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        assert len(config.dataset_attributes['articles']) > 0

        self.text_encoders = nn.ModuleDict({
            name:
            TextEncoder(word_embedding, config.word_embedding_dim,
                        config.num_filters, config.window_size,
                        config.query_vector_dim, config.dropout_probability)
            for name in (set(config.dataset_attributes['text_cols']))
        })

        category_embedding = nn.Embedding(config.num_categories,
                                          config.category_embedding_dim,
                                          padding_idx=0)
        self.element_encoders = nn.ModuleDict({
            name:
            CategoryEncoder(category_embedding, config.category_embedding_dim,
                           config.num_filters)
            for name in (set(config.dataset_attributes['category_cols']))
        })
        if len(config.dataset_attributes['articles']) > 1:
            self.final_attention = AdditiveAttention(config.query_vector_dim,
                                                     config.num_filters)

    def forward(self, articles):
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
        text_vectors = []
        for name, encoder in self.text_encoders.items():
            text_vectors.append(encoder(articles[name].to(device)))

        element_vectors = [
            encoder(articles[name].to(device))
            for name, encoder in self.element_encoders.items()
        ]

        all_vectors = text_vectors + element_vectors

        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:
            final_news_vector = self.final_attention(
                torch.stack(all_vectors, dim=1))
        return final_news_vector