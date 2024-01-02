import pandas
import gc

'''
This script reduces the memory usage of the raw data files.
The reduced data files are saved in the directory 'reduced_memory_data'.
By reduing the memory usage, some columns are converted to a different data type and with that, some information is lost.
To restore the original data, the following information is needed:
This is how to submit the predictions, assuming that the predictions are saved in a dataframe called PREDS_DF:
==================================>
    sub = pandas.read_csv('sample_submission.csv')[['customer_id']]
    sub['customer_id_2'] = sub['customer_id'].str[-16:].apply(lambda x: int(x, 16)).astype('int64')
    sub = sub.merge(PREDS_DF.rename({'customer_id':'customer_id_2'},axis=1), on='customer_id_2', how='left').fillna('')
    del sub['customer_id_2']
    sub.to_csv('submission.csv',index=False)
<===================================

To reverse the article_id conversion, the following information is needed:
==================================>
    articles_df['article_id'] = '0' + articles_df['article_id'].astype('str')
<====================================
'''

customers_path = 'raw_data/customers.csv'
articles_path = 'raw_data/articles.csv'
transactions_path = 'raw_data/transactions_train.csv'

save_dir = 'reduced_memory_data/'

# Reduce memory usage of customers.csv
print('Reducing memory usage of customers.csv...')
customers_df = pandas.read_csv(customers_path)
customers_df['customer_id'] = customers_df['customer_id'].str[-16:].apply(lambda x: int(x, 16)).astype('int64')
customers_df.to_csv(save_dir + 'customers.csv', index=False)
del customers_df
gc.collect()

# Reduce memory usage of articles.csv
print('Reducing memory usage of articles.csv...')
articles_df = pandas.read_csv(articles_path)
articles_df['article_id'] = articles_df['article_id'].astype('int32')
articles_df.to_csv(save_dir + 'articles.csv', index=False)
del articles_df
gc.collect()

# Reduce memory usage of transactions_train.csv
print('Reducing memory usage of transactions_train.csv...')
transactions_df = pandas.read_csv(transactions_path)
transactions_df['customer_id'] = transactions_df['customer_id'].str[-16:].apply(lambda x: int(x, 16)).astype('int64')
transactions_df['article_id'] = transactions_df['article_id'].astype('int32')
transactions_df['t_dat'] = pandas.to_datetime( transactions_df['t_dat'] )
transactions_df['price'] = transactions_df['price'].astype('float32')
transactions_df['sales_channel_id'] = transactions_df['sales_channel_id'].astype('int8')
transactions_df.to_csv(save_dir + 'transactions_train.csv', index=False)