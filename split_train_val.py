import warnings

import pandas as pd
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from datetime import timedelta


print('Load transactions_train.csv...')
transactions_df = pd.read_csv('data/transactions_train.csv')
# Convert timestamp value to week number since the start day of the dataset
transactions_df['week_number'] = pd.to_datetime(transactions_df['t_dat'])
transactions_df['week_number'] = (
    (transactions_df['week_number'] - transactions_df['week_number'].min() + timedelta(1)).dt.days
)
transactions_df['week_number'] = ((transactions_df['week_number'] - 1) // 7).astype('int8')

# Split train and validation dataset
print('Split train and validation dataset...')
last_week = transactions_df['week_number'].max()
train_df = transactions_df[transactions_df['week_number'] < last_week]
val_df = transactions_df[transactions_df['week_number'] == last_week]

train_df.drop(columns=['week_number'], inplace=True)
val_df.drop(columns=['week_number'], inplace=True)

# Save to csv
print('Save to csv...')
train_df.to_csv('processed_data/train/transactions_train.csv', index=False)
val_df.to_csv('processed_data/val/transactions_val.csv', index=False)