import sys
from os import path
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader

from config import NAMLConfig as config
from model.naml import NAML as Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def value2rank(d):
    values = list(d.values())
    ranks = [sorted(values, reverse=True).index(x) for x in values]
    return {k: ranks[i] + 1 for i, k in enumerate(d.keys())}


class ArticleDataset(Dataset):
    """Load articles for evaluation."""
    def __init__(self, articles_path):
        super(ArticleDataset, self).__init__()
        self.articles_parsed = pd.read_csv(
            articles_path,
            index_col='article_id',
            usecols=['article_id'] + config.dataset_attributes['articles'],
        )
        self.articles_parsed['detail_desc'] = self.articles_parsed['detail_desc'].apply(
            lambda x: [int(i) for i in x.split()[:config.num_words_detail_desc]] if isinstance(x, str) else x
        )
        self.articles2dict = self.articles_parsed.to_dict('index')
        for key1 in self.articles2dict.keys():
            for key2 in self.articles2dict[key1].keys():
                self.articles2dict[key1][key2] = torch.tensor(self.articles2dict[key1][key2])

    def __len__(self):
        return len(self.articles_parsed)

    def __getitem__(self, idx):
        item = self.articles2dict[idx]
        return item


class CustomerDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    """
    def __init__(self, customers_path, transactions_path, user2int_path):
        super(CustomerDataset, self).__init__()
        self.transactions_df = pd.read_csv(transactions_path)

        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user in user2int:
                self.behaviors.at[row.Index, 'user'] = user2int[row.user]
            else:
                user_missed += 1
                self.behaviors.at[row.Index, 'user'] = 0

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user":
            row.user,
            "clicked_news_string":
            row.clicked_news,
            "clicked_news":
            row.clicked_news.split()[:config.num_clicked_news_a_user]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = ['PADDED_ARTICLE'] * repeated_times + item["clicked_news"]
        return item


class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """
    def __init__(self, transactions_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(transactions_path,
                                       header=None,
                                       usecols=range(5),
                                       names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "impression_id": row.impression_id,
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions
        }
        return item


def calculate_single_user_metric(pair):
    try:
        auc = roc_auc_score(*pair)
        mrr = mrr_score(*pair)
        ndcg5 = ndcg_score(*pair, 5)
        ndcg10 = ndcg_score(*pair, 10)
        return [auc, mrr, ndcg5, ndcg10]
    except ValueError:
        return [np.nan] * 4


@torch.no_grad()
def evaluate(model, directory, num_workers, max_count=sys.maxsize):
    """
    Evaluate model on target directory.
    Args:
        model: model to be evaluated
        directory: the directory that contains two files (val/transactions_parsed.csv, articles_parsed.csv)
        num_workers: processes number for calculating metrics
    Returns:
        AUC
        MRR
        nDCG@5
        nDCG@10
    """
    articles_path = path.join(directory, 'articles_parsed.csv')
    transactions_path = path.join(directory, 'val', 'transactions_parsed.csv')
    user2int_path = path.join(directory, 'user2int.tsv')

    article_ds = ArticleDataset(articles_path)
    articles_dataloader = DataLoader(article_ds,
                                 batch_size=config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    article2vect = {}
    for minibatch in tqdm(articles_dataloader, desc="Calculating vectors for article"):
        articles_ids = minibatch["id"]
        if any(id not in article2vect for id in articles_ids):
            article_vectors = model.get_article_vector(minibatch)
            for id, vector in zip(articles_ids, article_vectors):
                if id not in article2vect:
                    article2vect[id] = vector

    article2vect['PADDED_ARTICLE'] = torch.zeros(list(article2vect.values())[0].size())

    user_dataset = CustomerDataset(transactions_path, user2int_path)
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    user2vector = {}
    for minibatch in tqdm(user_dataloader,
                          desc="Calculating vectors for users"):
        user_strings = minibatch["clicked_news_string"]
        if any(user_string not in user2vector for user_string in user_strings):
            clicked_news_vector = torch.stack([
                torch.stack([article2vect[x].to(device) for x in news_list],
                            dim=0) for news_list in minibatch["clicked_news"]
            ], dim=0).transpose(0, 1)
            user_vector = model.get_user_vector(clicked_news_vector)
            for user, vector in zip(user_strings, user_vector):
                if user not in user2vector:
                    user2vector[user] = vector

    behaviors_dataset = BehaviorsDataset(path.join(directory, 'behaviors.tsv'))
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=config.num_workers)

    count = 0

    tasks = []

    for minibatch in tqdm(behaviors_dataloader,
                          desc="Calculating probabilities"):
        count += 1
        if count == max_count:
            break

        candidate_news_vector = torch.stack([
            article2vect[news[0].split('-')[0]]
            for news in minibatch['impressions']
        ],
                                            dim=0)
        user_vector = user2vector[minibatch['clicked_news_string'][0]]
        click_probability = model.get_prediction(candidate_news_vector,
                                                 user_vector)

        y_pred = click_probability.tolist()
        y_true = [
            int(news[0].split('-')[1]) for news in minibatch['impressions']
        ]

        tasks.append((y_true, y_pred))

    with Pool(processes=num_workers) as pool:
        results = pool.map(calculate_single_user_metric, tasks)

    aucs, mrrs, ndcg5s, ndcg10s = np.array(results).T
    return np.nanmean(aucs), np.nanmean(mrrs), np.nanmean(ndcg5s), np.nanmean(
        ndcg10s)


if __name__ == '__main__':
    print('Using device:', device)
    print(f'Evaluating model NAML')
    # Don't need to load pretrained word/entity/context embedding
    # since it will be loaded from checkpoint later
    model = Model(config).to(device)
    from train import latest_checkpoint  # Avoid circular imports
    checkpoint_path = latest_checkpoint(path.join('./checkpoint', 'NAML'))
    if checkpoint_path is None:
        print('No checkpoint file found!')
        exit()
    print(f"Load saved parameters in {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    auc, mrr, ndcg5, ndcg10 = evaluate(model, './data/test',
                                       config.num_workers)
    print(
        f'AUC: {auc:.4f}\nMRR: {mrr:.4f}\nnDCG@5: {ndcg5:.4f}\nnDCG@10: {ndcg10:.4f}'
    )