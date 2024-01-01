import warnings

import pandas as pd
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from datetime import timedelta
from tqdm import tqdm

from src.config import NAMLConfig as config


print('Load transactions_train.csv...')
transactions_df = pd.read_csv('data/transactions_train.csv')
transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'])
transactions_df = transactions_df[transactions_df['t_dat'] >= '2019-09-01']
transactions_df.to_csv('data/transactions_train.csv', index=False)

new_transactions_df = transactions_df.groupby(['t_dat', 'customer_id']).agg(
    article_id = ('article_id', list),
).reset_index().sort_values(by=['t_dat', 'customer_id']).reset_index(drop=True)
new_transactions_df['prev_purchased'] = None
prev_purchases_dict = {}
for idx, row in tqdm(new_transactions_df.iterrows(), total=len(new_transactions_df)):
    if row['customer_id'] in prev_purchases_dict:
        prev_purchases = prev_purchases_dict[row['customer_id']]
    else:
        prev_purchases = []
    new_transactions_df.at[idx, 'prev_purchased'] = prev_purchases.copy()
    for article_id in row['article_id']:
        if len(prev_purchases) >= config.max_history:
            prev_purchases.pop(0)
        prev_purchases.append(article_id)
    if len(prev_purchases) > config.max_history:
        prev_purchases = prev_purchases[-config.max_history:]
    prev_purchases_dict[row['customer_id']] = prev_purchases
    
new_transactions_df = new_transactions_df.explode('article_id')
new_transactions_df.to_csv('data/transactions_parsed.csv', index=False)

# Convert timestamp value to week number since the start day of the dataset
new_transactions_df['week_number'] = pd.to_datetime(new_transactions_df['t_dat'])
new_transactions_df['week_number'] = (
    (new_transactions_df['week_number'] - new_transactions_df['week_number'].min() + timedelta(1)).dt.days
)
new_transactions_df['week_number'] = ((new_transactions_df['week_number'] - 1) // 7).astype('int8')

# Split train and validation dataset
print('Split train and validation dataset...')
last_week = new_transactions_df['week_number'].max()
train_df = new_transactions_df[new_transactions_df['week_number'] < last_week]
val_df = new_transactions_df[new_transactions_df['week_number'] == last_week]

train_df.drop(columns=['week_number'], inplace=True)
val_df.drop(columns=['week_number'], inplace=True)

# Save to csv
print('Save to csv...')
train_df.to_csv('processed_data/train/transactions_parsed.csv', index=False)
val_df.to_csv('processed_data/val/transactions_parsed.csv', index=False)