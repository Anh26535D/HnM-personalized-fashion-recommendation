import sys
from os import path
from ast import literal_eval

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from config import NAMLConfig as config
from model.naml import NAML as Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseDataset(Dataset):
    def __init__(self, transactions_path, articles_path):
        super(BaseDataset, self).__init__()

        self.transactions_parsed = pd.read_csv(transactions_path)
        self.transactions_parsed = self.transactions_parsed.groupby(['customer_id']).agg(
            article_ids = ('article_id', list),
            prev_purchased = ('prev_purchased', 'first'),
        ).reset_index()
        self.temp_transactions = self.transactions_parsed['article_ids'].copy()
        self.transactions_parsed = self.transactions_parsed.head(10)

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
                
        padding_all = {k: 0 for k in config.dataset_attributes['articles']}
        padding_all['detail_desc'] = [0] * config.num_words_detail_desc

        for key in padding_all.keys():
            padding_all[key] = torch.tensor(padding_all[key])

        self.padding = {
            k: v for k, v in padding_all.items()
            if k in config.dataset_attributes['articles']
        }
        self.artice_ids = self.articles_parsed.index.tolist()
        self.articles_id2int = {x: i for i, x in enumerate(self.artice_ids)}
        self.int2articles_id = {i: x for i, x in enumerate(self.artice_ids)}
        self.prepared_candidate = [
            self.articles2dict[x] for x in tqdm(self.artice_ids, desc="Loading candidate articles")
        ]

    def __len__(self):
        return len(self.transactions_parsed)

    def __getitem__(self, idx):
        row = self.transactions_parsed.iloc[idx]
        item = {}
        item["article_ids"] = literal_eval(row['article_ids']) if isinstance(row['article_ids'], str) else row['article_ids']

        prev_purchased = literal_eval(row['prev_purchased'])
        if len(prev_purchased) > config.num_prev_purchased:
            prev_purchased = prev_purchased[-config.num_prev_purchased:]

        item["prev_purchased_parsed"] = [self.articles2dict[x] for x in prev_purchased]
        repeated_times = config.num_prev_purchased - len(item["prev_purchased_parsed"])
        assert repeated_times >= 0
        item["prev_purchased_parsed"] = [self.padding] * repeated_times + item["prev_purchased_parsed"]

        for key in config.dataset_attributes['record']:
            item[key] = row[key]

        item["candidate_articles"] = self.prepared_candidate
        return item

def apk_score(pair, k=12):
    """Calculate Average Precision at K (AP@K)"""
    actual, predicted = pair
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def cal_metrics(pair):
    actuals, predicteds = pair
    recalls = []
    apks = []
    for actual, predicted in zip(actuals, predicteds):
        if len(actual) == 0:
            continue
        TP = len(set(actual) & set(predicted))
        FN = len(set(actual) - set(predicted))
        if TP + FN == 0:
            return [np.nan, np.nan]
        recall = TP / (TP + FN)
        apk = apk_score((actual, predicted))
        recalls.append(recall)
        apks.append(apk)

    return [np.mean(recall), np.mean(apks)]


@torch.no_grad()
def evaluate(model, directory, max_count=sys.maxsize):
    """
    Evaluate model on target directory.
    Args:
        model: model to be evaluated
        directory: the val directory
    Returns:
        Recall
        APK@12
    """
    dataset = BaseDataset(path.join(directory, 'val', 'transactions_parsed.csv'),
                          path.join(directory, 'articles_parsed.csv'))
    print(f"Load valid dataset with size {len(dataset)}.")
    dataloader = iter(DataLoader(dataset,
                                batch_size=1,
                                num_workers=config.num_workers,
                                drop_last=True,
                                pin_memory=True))

    count = 0
    tasks = []
    candidate_articles_g = None
    for minibatch in tqdm(dataloader, desc="Calculating probabilities"):
        count += 1
        if count == max_count:
            break

        candidates = minibatch["candidate_articles"]
        prev_purchased = minibatch["prev_purchased_parsed"]

        if candidate_articles_g is None:
            candidate_articles = torch.stack([
                model.get_article_vector(article) for article in tqdm(candidates, desc="Loading candidate articles vector")
            ], dim=1)
            candidate_articles_g = candidate_articles
        else:
            candidate_articles = candidate_articles_g
        bought_articles = torch.stack([
            model.get_article_vector(x) for x in prev_purchased
        ], dim=1)
        user_vector = model.get_customer_vector(bought_articles)

        click_probability = torch.bmm(candidate_articles, user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)

        # Shape of batch_size * num_candidates
        y_pred = click_probability
        y_pred = F.softmax(y_pred, dim=1)
        # We just use top 1000 articles for calculating recall
        y_pred_topk = y_pred.topk(k=1000, dim=1).indices
        y_pred_topk = y_pred_topk.tolist()
        y_pred_topk = [[dataset.int2articles_id[x] for x in sample] for sample in y_pred_topk]

        y_true = minibatch['article_ids'] # batch_size
        y_true = [sample.tolist() for sample in y_true]
        
        tasks.append((y_true, y_pred_topk))

    results = list(map(cal_metrics, tasks))
    recalls, apks = np.array(results).T
    return np.nanmean(recalls), np.nanmean(apks)


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
    print("test model")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    recall, apk12 = evaluate(model, r'E:\project_deep_learning\personalized_fashion_recommendation\processed_data')
    print(
        f'Recall: {recall:.4f}\nAPK@12: {apk12:.4f}\n'
    )