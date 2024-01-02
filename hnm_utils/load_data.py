import pandas as pd

customers_dtypes = {
    "FN": "category",
    "Active": "category",
    "club_member_status": "category",
    "fashion_news_frequency": "category",
    "age": "float32",
    "postal_code": "category",
}

articles_dtypes = {
    "prod_name": "category",
    "product_type_name": "category",
    "product_group_name": "category",
    "graphical_appearance_name": "category",
    "colour_group_name": "category",
    "perceived_colour_value_name": "category",
    "perceived_colour_master_name": "category",
    "department_name": "category",
    "garment_group_name": "category",
    "section_name": "category",
    "index_group_name": "category",
    "index_code": "category",
    "index_name": "category",
    "detail_desc": "category",
    "product_code": "int32",
    "product_type_no": "int16",
    "graphical_appearance_no": "int32",
    "colour_group_code": "int8",
    "perceived_colour_value_id": "int8",
    "perceived_colour_master_id": "int8",
    "department_no": "int16",
    "index_group_no": "int8",
    "section_no": "int8",
    "garment_group_no": "int16",
}

def load_customers(data_path):
    customers_df = pd.read_csv(data_path, dtype=customers_dtypes)
    return customers_df

def load_articles(data_path):
    articles_df = pd.read_csv(data_path, dtype=articles_dtypes)
    return articles_df

def load_transactions(data_path):
    transactions_df = pd.read_csv(data_path)
    return transactions_df

def load_submission(data_path):
    submission_df = pd.read_csv(data_path)
    return submission_df