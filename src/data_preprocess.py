import os
import csv
import string

import numpy as np
import pandas as pd
import nltk
import swifter
from tqdm import tqdm
from nltk.tokenize import word_tokenize

nltk.download('punkt')

from config import NAMLConfig as config


def parse_customers(source, target, user2int_path):
    """Parse customers file.

    Args:
        source: source customers file
        target: target customers file
        user2int_path: path for saving user2int file
    """
    print(f"Parse {source}")

    customers_df = pd.read_csv(source)
    customer_ids = customers_df['customer_id'].unique()
    user2int = {id: i + 1 for i, id in enumerate(customer_ids)} # 0 for UNK

    pd.DataFrame(user2int.items(), columns=['user','int']) \
            .to_csv(user2int_path, sep='\t', index=False)
    print(f'Please modify `num_users` in `src/config.py` into 1 + {len(user2int)}')

    customers_df['customer_id'] = customers_df['customer_id'].map(user2int)
    customers_df.to_csv(target, index=False)


def parse_articles(source, target, category2int_path, word2int_path):
    """Parse articles file.

    Args:
        source: source articles file
        target: target articles file
        category2int_path, word2int_path: Path for saving to
    """
    print(f"Parse {source}")
    cat_cols = [
        'prod_name',
        'product_type_name',
        'product_group_name',
        'graphical_appearance_name',
        'colour_group_name',
        'perceived_colour_value_name',
        'perceived_colour_master_name',
        'department_name',
        'index_name',
        'index_group_name',
        'section_name',
        'garment_group_name',
    ]
    text_cols =  ['detail_desc']
    articles_df = pd.read_csv(source, usecols=['article_id'] + cat_cols + text_cols)
    articles_df.fillna(' ', inplace=True)
    def remove_punctuation(input_text):
        input_text = str(input_text).lower()
        translation_table = str.maketrans("", "", string.punctuation)
        return input_text.translate(translation_table)
    articles_df['detail_desc'] = articles_df['detail_desc'].swifter.apply(remove_punctuation)

    def parse_row(row):
        new_row = {}
        new_row['article_id'] = row['article_id']
        for col in cat_cols:
            new_row[col] = category2int[row[col]] if row[col] in category2int else 0

        for col in text_cols:
            tokenized = [str(word2int[w]) for w in word_tokenize(row[col]) if w in word2int]
            if len(tokenized) < config.num_words_detail_desc:
                tokenized += ['0'] * (config.num_words_detail_desc - len(tokenized))
            new_row[col] = ' '.join(tokenized)

        return pd.Series(new_row)

    category2int = {}
    word2int = {}
    word2freq = {}

    for idx, row in articles_df.iterrows():
        for col in cat_cols:
            if row[col] not in category2int:
                category2int[row[col]] = len(category2int) + 1

        for col in text_cols:
            for w in word_tokenize(row[col]):
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1

    for k, v in word2freq.items():
        if v >= config.word_freq_threshold:
            word2int[k] = len(word2int) + 1

    parsed_articles = articles_df.swifter.apply(parse_row, axis=1)
    parsed_articles.to_csv(target, index=False)

    pd.DataFrame(category2int.items(), columns=['category', 'int']) \
            .to_csv(category2int_path, sep='\t', index=False)
    print(f'Please modify `num_categories` in `src/config.py` into 1 + {len(category2int)}')

    pd.DataFrame(word2int.items(), columns=['word', 'int']) \
            .to_csv(word2int_path, sep='\t', index=False)
    print(f'Please modify `num_words` in `src/config.py` into 1 + {len(word2int)}')


def parse_transactions(source, target, user2int_path):
    """Parse transactions file.
    
    Args:
        source: source transactions file
        target: target transactions file
        user2int_path: path for loading user2int file
    """
    print(f"Parse {source}")

    transactions_df = pd.read_csv(source)
    user2int = pd.read_csv(user2int_path, sep='\t', index_col='user')['int'].to_dict()
    transactions_df['customer_id'] = transactions_df['customer_id'].map(user2int)

    transactions_df.to_csv(target, index=False)


def generate_word_embedding(source, target, word2int_path):
    """
    Generate from pretrained word embedding file
    If a word not in embedding file, initial its embedding by N(0, 1)
    Args:
        source: path of pretrained word embedding file, e.g. glove.6B.100d.txt
        target: path for saving word embedding. Will be saved in numpy format
        word2int_path: vocabulary file when words in it will be searched in pretrained embedding file
    """
    # na_filter=False is needed since nan is also a valid word
    # word, int
    word2int = pd.read_csv(word2int_path, na_filter=False, sep='\t').set_index('word')
    source_embedding = pd.read_csv(source,
                                     index_col=0,
                                     sep=' ',
                                     header=None,
                                     quoting=csv.QUOTE_NONE,
                                     names=range(config.word_embedding_dim))
    source_embedding.index.rename('word', inplace=True)
    # word, int, vector
    merged = word2int.merge(source_embedding,
                            how='inner',
                            left_index=True,
                            right_index=True)
    merged.set_index('int', inplace=True)

    missed_index = np.setdiff1d(np.arange(len(word2int) + 1), merged.index.values)
    missed_embedding = pd.DataFrame(data=np.random.normal(
        size=(len(missed_index), config.word_embedding_dim)))
    missed_embedding['int'] = missed_index
    missed_embedding.set_index('int', inplace=True)
    final_embedding = pd.concat([merged, missed_embedding]).sort_index()
    np.save(target, final_embedding.values)

    print(f'Rate of word missed in pretrained embedding: {(len(missed_index)-1)/len(word2int):.4f}')

if __name__ == '__main__':
    raw_data = './data'
    processed_data_dir = './processed_data'
    train_dir = os.path.join(processed_data_dir, 'train')
    val_dir = os.path.join(processed_data_dir, 'val')
    word_embedding_path = os.path.join(processed_data_dir, 'glove', f'glove.6B.{config.word_embedding_dim}d.txt')

    print('Process data for training')

    print('Parse customers')
    parse_customers(os.path.join(raw_data, 'customers.csv'),
                    os.path.join(processed_data_dir, 'customers_parsed.csv'),
                    os.path.join(processed_data_dir, 'user2int.csv'))
    
    print('Parse articles')
    parse_articles(os.path.join(raw_data, 'articles.csv'),
                    os.path.join(processed_data_dir, 'articles_parsed.csv'),
                    os.path.join(processed_data_dir, 'category2int.csv'),
                    os.path.join(processed_data_dir, 'word2int.csv'))

    print('Parse transactions')
    parse_transactions(os.path.join(train_dir, 'transactions_parsed.csv'),
                    os.path.join(train_dir, 'transactions_parsed.csv'),
                    os.path.join(processed_data_dir, 'user2int.csv'))

    print('\nProcess data for validation')

    print('Parse transactions')
    parse_transactions(os.path.join(val_dir, 'transactions_parsed.csv'),
                    os.path.join(val_dir, 'transactions_parsed.csv'),
                    os.path.join(processed_data_dir, 'user2int.csv'))

    print('Generate word embedding')
    generate_word_embedding(word_embedding_path,
                        os.path.join(processed_data_dir, 'pretrained_word_embedding.npy'),
                        os.path.join(processed_data_dir, 'word2int.csv'))

