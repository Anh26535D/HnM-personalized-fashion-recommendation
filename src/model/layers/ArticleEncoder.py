from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.layers.AdditiveAttention import AdditiveAttention


class TextEncoder(nn.Module):
    def __init__(
        self,
        tokenizer,
        word_embedding,
        embed_dim,
        max_words_per_text,
        num_filters,
        window_size,
        query_vector_dim,
        dropout_proba,
        is_training,
    ):
        super(TextEncoder, self).__init__()

        self.tokenizer = tokenizer
        self.word_embedding = word_embedding
        self.embed_dim = embed_dim
        self.max_words_per_text = max_words_per_text
        self.num_filters = num_filters
        self.window_size = window_size
        self.query_vector_dim = query_vector_dim
        self.dropout_proba = dropout_proba
        self.is_training = is_training

        self.cnn_encode = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(window_size, embed_dim),
            padding=(int((window_size - 1) / 2), 0)
        )

        self.additive_attention = AdditiveAttention(
            candidate_vector_dim=num_filters,
            query_vector_dim=query_vector_dim
        )

    def forward(self, articles):
        # articles has shape of [num_articles, len(articles.columns)]
        X = [self.tokenizer(text) for text in articles]
        X = [tokens+[""] * (self.max_words_per_text-len(tokens)) if len(tokens) <
             self.max_words_per_text else tokens[:self.max_words_per_text] for tokens in X]

        # X_tensor has shape of [num_articles, max_word_per_text, word_embed_dim]
        X_tensor = torch.zeros(
            len(articles), self.max_words_per_text, self.embed_dim)
        for i, tokens in enumerate(X):
            X_tensor[i] = self.word_embedding.get_vecs_by_tokens(tokens)

        # conv_encoded_X has shape of [num_articles, num_filters, max_word_per_text-1]
        conv_encoded_X = self.cnn_encode(
            X_tensor.unsqueeze(dim=1)).squeeze(dim=3)
        conv_encoded_X = F.dropout(F.relu(conv_encoded_X),
                                   p=self.dropout_proba,
                                   training=self.is_training)

        # conv_encoded_X has shape of [num_articles, max_word_per_text-1, num_filters]
        conv_encoded_X = conv_encoded_X.transpose(1, 2)

        # text_vector has shape of [batch_size, num_filters]
        text_vector = self.additive_attention(conv_encoded_X)
        return text_vector


class CategoryEncoder(nn.Module):
    def __init__(self, embedding, linear_input_dim, linear_output_dim):
        super(CategoryEncoder, self).__init__()
        self.embedding = embedding
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)

    def forward(self, element):
        return F.relu(self.linear(self.embedding(element)))


class ArticleEncoder(nn.Module):
    def __init__(
            self, 
            text_cols: List, 
            category_cols: List,
            # For TextEncoder
            tokenizer,
            pretrained_word_embedding,
            word_embedding_dim,
            max_words_per_text,
            query_vector_dim,
            num_filters,
            window_size,
            dropout_proba,
            is_training,
            # For CategoryEncoder
            category_embedding_dim,
            num_categories, # list of number of categories for each category column 
        ):
        super(ArticleEncoder, self).__init__()

        self.text_encoders = nn.ModuleDict({
            name:
            TextEncoder(
                tokenizer,
                pretrained_word_embedding, 
                word_embedding_dim,
                max_words_per_text,
                num_filters, 
                window_size,
                query_vector_dim,
                dropout_proba,
                is_training
            ) for name in text_cols
        })

        self.element_encoders = nn.ModuleDict({
            name:
            CategoryEncoder(
                nn.Embedding(num_categories[idx], category_embedding_dim, padding_idx=0), 
                category_embedding_dim,
                num_filters
            ) for idx, name in enumerate(category_cols)
        })

        self.final_attention = AdditiveAttention(
            candidate_vector_dim=num_filters,
            query_vector_dim=query_vector_dim, 
        )

    def forward(self, articles):
        text_vectors = [
            encoder(articles[name])
            for name, encoder in self.text_encoders.items()
        ]
        element_vectors = [
            encoder(torch.tensor(articles[name]))
            for name, encoder in self.element_encoders.items()
        ]

        all_vectors = text_vectors + element_vectors
        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:
            final_news_vector = self.final_attention(
                torch.stack(all_vectors, dim=1))
        return final_news_vector