from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from ast import literal_eval

from config import NAMLConfig as config


class BaseDataset(Dataset):
    def __init__(self, transactions_path, articles_path):
        super(BaseDataset, self).__init__()

        assert all(attribute in [
            'prod_name', 'product_type_name', 'product_group_name', 
            'graphical_appearance_name', 'colour_group_name', 
            'perceived_colour_value_name', 'perceived_colour_master_name', 
            'department_name', 'index_name', 'index_group_name', 
            'section_name', 'garment_group_name', 'detail_desc'
        ] for attribute in config.dataset_attributes['articles'])

        assert all(attribute in ['customer_id']
                   for attribute in config.dataset_attributes['record'])

        self.transactions_parsed = pd.read_csv(transactions_path)

        self.articles_parsed = pd.read_csv(
            articles_path,
            index_col='article_id',
            usecols=['article_id'] + config.dataset_attributes['articles'],
        )

        self.articles_parsed['detail_desc'] = self.articles_parsed['detail_desc'].apply(
            lambda x: [int(i) for i in x.split()[:config.num_words_detail_desc]] if isinstance(x, str) else x
        )
        
        self.articles_id2int = {x: i for i, x in enumerate(self.articles_parsed.index)}
        self.articles2dict = self.articles_parsed.to_dict('index')

        for key1 in self.articles2dict.keys():
            for key2 in self.articles2dict[key1].keys():
                self.articles2dict[key1][key2] = torch.tensor(self.articles2dict[key1][key2])
                
        padding_all = {k: 0 for k in config.dataset_attributes['articles']}
        padding_all['detail_desc'] = [0] * config.num_words_detail_desc

        for key in padding_all.keys():
            padding_all[key] = torch.tensor(padding_all[key])

        self.padding = {
            k: v for k, v in padding_all.items()
            if k in config.dataset_attributes['articles']
        }
        self.artice_ids = self.articles_parsed.index.tolist()

    def __len__(self):
        return len(self.transactions_parsed)

    def __getitem__(self, idx):
        row = self.transactions_parsed.iloc[idx]
        item = {}
        item["article_id"] = row['article_id']
        item['article_id_parsed'] = [self.articles2dict[row['article_id']]]

        prev_purchased = literal_eval(row['prev_purchased'])
        if len(prev_purchased) > config.num_prev_purchased:
            prev_purchased = prev_purchased[-config.num_prev_purchased:]

        item["prev_purchased_parsed"] = [self.articles2dict[x] for x in prev_purchased]
        repeated_times = config.num_prev_purchased - len(item["prev_purchased_parsed"])
        assert repeated_times >= 0
        item["prev_purchased_parsed"] = [self.padding] * repeated_times + item["prev_purchased_parsed"]

        for key in config.dataset_attributes['record']:
            item[key] = row[key]

        np.random.shuffle(self.artice_ids)
        item["candidate_articles"] = [
            self.articles2dict[x] for x in self.artice_ids[:config.num_random_sampled_articles]
        ]
        item["candidate_articles"] += item['article_id_parsed']
        return item