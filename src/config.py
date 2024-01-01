class BaseConfig():
    """
    General configurations
    """
    num_epochs = 1
    num_batches_show_loss = 10  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 100
    batch_size = 128
    learning_rate = 0.01
    num_workers = 0  # Number of workers for data loading, in windows, it should be 0 (bug)
    num_prev_purchased = 30  # Number of sampled purchase history for each user
    num_words_detail_desc = 100  # Number of words in detail description
    word_freq_threshold = 1
    negative_sampling_ratio = 100  # K
    num_random_sampled_articles = 1  # Number of random sampled articles for each user
    dropout_probability = 0.2
    # Modify the following by the output of `src/data_preprocess.py`
    max_history = 30  # Max number of purchase history stored for each user
    processed_data_path = "processed_data/"
    num_words = 1 + 4889
    num_categories = 1 + 46411
    num_users = 1 + 1371980
    word_embedding_dim = 100
    category_embedding_dim = 100



class NAMLConfig(BaseConfig):
    dataset_attributes = {
        "articles": [
            'prod_name', 'product_type_name', 'product_group_name', 
            'graphical_appearance_name', 'colour_group_name', 'perceived_colour_value_name', 
            'perceived_colour_master_name', 'department_name', 'index_name', 'index_group_name', 
            'section_name', 'garment_group_name', 'detail_desc'
        ],
        "record": ['customer_id'],
        "text_cols": ['detail_desc'], 
        "category_cols": [
            'prod_name', 'product_type_name', 'product_group_name', 
            'graphical_appearance_name', 'colour_group_name', 'perceived_colour_value_name', 
            'perceived_colour_master_name', 'department_name', 'index_name', 'index_group_name', 
            'section_name', 'garment_group_name'
        ],
    }
    # For CNN
    num_filters = 300
    window_size = 3
    # For additive attention
    query_vector_dim = 200