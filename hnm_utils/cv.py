from typing import List

import pandas as pd


def ground_truth(transactions_df: pd.DataFrame):
    """Get list of bought articles for each customer."""
    gt = transactions_df[["customer_id", "article_id"]].drop_duplicates()
    gt = gt.groupby("customer_id")[["article_id"]].agg(list)
    gt.columns = ["prediction"]
    gt.reset_index(inplace=True)
    return gt


def feature_label_split(transactions_df: pd.DataFrame, week_label: int, week_length: int = None):
    """Split transaction_df into train_df and label_df based on week_number."""

    label_df = transactions_df.query("week_number == @week_label")
    features_df = transactions_df.query("week_number < @week_label")
    if week_length is not None:
        features_df = features_df.query(f"week_number >= {week_label - week_length}")
    return features_df, label_df


def report_candidates(candidates: pd.DataFrame, actual_articles: pd.DataFrame):
    """Calculate number of hit candidates, ignoring the order of articles."""
    
    num_candidates = len(candidates)
    num_actual_articles = len(actual_articles)
    union = pd.concat([candidates, actual_articles]).drop_duplicates()
    num_hit_candidates = num_candidates + num_actual_articles - len(union)
    
    print("######################### Retrieval Report ##############################")
    print(f"Number of hit candidates: {num_hit_candidates}")
    print(f"Number of actual articles: {num_actual_articles}")
    print(f"Number of candidates: {num_candidates}")
    print(f"Precision: {num_hit_candidates / num_candidates}")
    print(f"Recall: {num_hit_candidates / num_actual_articles}")
    print("########################################################################")


def cal_apk(actual, predict, len_predict, len_actual, k):
    """Calculate average precision at k"""
    num_hits = 0
    score = 0.0

    for i in range(len_predict):
        if predict[i] in actual:
            num_hits += 1
            score += num_hits / (i + 1)

    return score / min(len_actual, k)
    

def cal_mapk(actual: pd.Series, predict: pd.Series, k: int = 12) -> float:
    """Calculate mean average precision at k"""
    actual = actual.apply(list)
    predict = predict.apply(list)

    # we don't score ones without any purchases
    actual = actual[actual.notna()]
    actual = actual[actual.apply(len) > 0]

    if len(actual) == 0:
        raise ValueError("actual is empty")

    eval_df = pd.DataFrame({"actual": actual})
    eval_df["len_actual"] = eval_df["actual"].apply(len)

    eval_df["predict"] = predict
    eval_df["predict"] = eval_df["predict"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    eval_df["len_predict"] = eval_df["predict"].apply(lambda x: min(len(x), 12))
    eval_df = eval_df[eval_df["len_predict"] > 0]

    eval_df["score"] = eval_df.apply(
        lambda x: cal_apk(
            x["actual"], x["predict"], x["len_predict"], x["len_actual"], k=k
        ),
        axis=1,
    )

    return eval_df["score"].sum() / len(actual)