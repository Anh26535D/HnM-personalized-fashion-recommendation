import copy
from datetime import datetime
from typing import List

import pandas as pd
import numpy as np
import lightgbm
from lightgbm import LGBMRanker
from sklearn.metrics import roc_auc_score


def create_predictions(ids_df, preds):
    ids_df["pred"] = preds
    ids_df = ids_df.sort_values(["customer_id", "pred"], ascending=False)
    ids_df = ids_df.groupby("customer_id").head(12)
    predictions = ids_df.groupby("customer_id")["article_id"].agg(list)
    return predictions


def pred_in_batches(model, features_df, batch_size=1000000):
    preds = []
    num_batches = int(len(features_df) / batch_size) + 1
    for batch in range(num_batches):
        batch_df = features_df.iloc[batch * batch_size : (batch + 1) * batch_size, :]
        preds.append(model.predict(batch_df))

    return np.concatenate(preds)

def prepare_dfs_for_model(
        transactions_df: pd.DataFrame, 
        customers_df: pd.DataFrame, 
        articles_df: pd.DataFrame, 
        generate_candidates_func: callable, 
        **config
    ):
    """Prepare dataframes for model"""
    candidates_df, label_df = generate_candidates_func(transactions_df, customers_df, articles_df, **config)

    # adding y column
    label_df = label_df[["customer_id", "article_id"]].drop_duplicates()
    label_df["match"] = 1
    candidates_df = candidates_df.merge(label_df, how="left", on=["customer_id", "article_id"])
    candidates_df["match"] = candidates_df["match"].fillna(0).astype("int8")
    label_df.drop("match", axis=1, inplace=True)

    # remove customers with no positives (no purchases in label week)
    customers_with_positives = candidates_df.query("match==1")["customer_id"].unique()
    candidates_df = candidates_df[candidates_df["customer_id"].isin(customers_with_positives)]
    candidates_df = candidates_df.sort_values("customer_id").reset_index(drop=True)

    # get group lengths
    group_lengths = list(candidates_df.groupby("customer_id")["article_id"].count())

    # get query ids (qids)
    qids = candidates_df[["customer_id", "article_id"]].copy()
    candidates_df.drop(["customer_id", "article_id"], axis=1, inplace=True)

    y = candidates_df["match"]
    X = candidates_df.drop("match", axis=1)

    return qids, X, group_lengths, y, label_df


def prepare_concat_train_modeling_dfs(t, c, a, cand_features_func, **params):
    working_params = copy.deepcopy(params)

    if params["num_concats"] > 1:
        empty_list = [None] * params["num_concats"]
        empty_lists = [empty_list.copy() for i in range(5)]

        (
            train_ids_df,
            train_X,
            train_group_lengths,
            train_y,
            train_truth_df,
        ) = empty_lists

        for i in range(params["num_concats"]):
            week_num = params["label_week"] - (i + 1)
            print(f"preparing training modeling dfs for {week_num}...")
            working_params["label_week"] = week_num
            (
                train_ids_df[i],
                train_X[i],
                train_group_lengths[i],
                train_y[i],
                train_truth_df[i],
            ) = prepare_dfs_for_model(t, c, a, cand_features_func, **working_params)

        print("concatenating all weeks together")
        train_ids_df = pd.concat(train_ids_df)
        train_group_lengths = sum(train_group_lengths, [])
        train_truth_df = pd.concat(train_truth_df)
        train_X = pd.concat(train_X)
        train_y = pd.concat(train_y)
    else:
        week_num = params["label_week"] - 1
        print(f"preparing training modeling dfs for {week_num}...")
        working_params["label_week"] = week_num
        (
            train_ids_df,
            train_X,
            train_group_lengths,
            train_y,
            train_truth_df,
        ) = prepare_dfs_for_model(t, c, a, cand_features_func, **working_params)

    return train_ids_df, train_X, train_group_lengths, train_y, train_truth_df


def prepare_train_eval_modeling_dfs(t, c, a, cand_features_func, **params):
    (
        train_ids_df,
        train_X,
        train_group_lengths,
        train_y,
        train_truth_df,
    ) = prepare_concat_train_modeling_dfs(t, c, a, cand_features_func, **params)

    print("preparing evaluation modeling dfs...")
    (
        eval_ids_df,
        eval_X,
        eval_group_lengths,
        eval_y,
        eval_truth_df,
    ) = prepare_dfs_for_model(t, c, a, cand_features_func, **params)

    return (
        train_ids_df,
        train_X,
        train_group_lengths,
        train_y,
        train_truth_df,
        eval_ids_df,
        eval_X,
        eval_group_lengths,
        eval_y,
        eval_truth_df,
    )


def full_cv_run(t, c, a, cand_features_func, score_func, **kwargs):
    # load training data
    (
        train_ids_df,
        train_X,
        train_group_lengths,
        train_y,
        train_truth_df,
        eval_ids_df,
        eval_X,
        eval_group_lengths,
        eval_y,
        eval_truth_df,
    ) = prepare_train_eval_modeling_dfs(
        t, c, a, cand_features_func, **kwargs
    )

    model = LGBMRanker(**(kwargs["lgbm_params"]))

    eval_set = [(train_X, train_y), (eval_X, eval_y)]
    eval_group = [train_group_lengths, eval_group_lengths]
    eval_names = ["train", "validation"]

    le_callback = lightgbm.log_evaluation(kwargs["log_evaluation"])
    es_callback = lightgbm.early_stopping(kwargs["early_stopping"])

    model.fit(
        train_X,
        train_y,
        eval_set=eval_set,
        eval_names=eval_names,
        eval_group=eval_group,
        eval_metric="MAP",
        eval_at=kwargs["eval_at"],
        callbacks=[le_callback, es_callback],
        group=train_group_lengths,
    )

    # get feature importance before save/reloading model
    feature_importance_dict = dict(
        zip(list(eval_X.columns), list(model.feature_importances_))
    )
    feature_importance_series = pd.Series(feature_importance_dict).sort_values()

    # save/reload model
    model_path = f"model_{kwargs['label_week']}"
    model.booster_.save_model(model_path)
    model = lightgbm.Booster(model_file=model_path)
    # train predictions and scores
    train_pred = model.predict(train_X)
    print("Train AUC {:.4f}".format(roc_auc_score(train_y, train_pred)))
    print("Train score: ", score_func(train_ids_df, train_pred, train_truth_df))

    del train_X, train_y, train_group_lengths, train_truth_df, train_pred

    # evaluation predictions and scores
    eval_pred = pred_in_batches(model, eval_X)

    print("Eval AUC {:.4f}".format(roc_auc_score(eval_y, eval_pred)))
    eval_score = score_func(eval_ids_df, eval_pred, eval_truth_df)
    print("Eval score:", eval_score)

    # print feature importances
    print("\n")
    print(feature_importance_series)

    return eval_score


def run_all_cvs(
        transactions_df: pd.DataFrame, 
        customers_df: pd.DataFrame, 
        articles_df: pd.DataFrame, 
        generate_candidates_func: callable, 
        model_report_func: callable, 
        cv_weeks: List[int] = [102, 103, 104], 
        **params
    ):
    """Run cross validation and return average score"""
    cv_scores = []
    total_duration = datetime.now() - datetime.now()

    for cv_week in cv_weeks:
        starting_time = datetime.now()

        cv_params = copy.deepcopy(params)
        cv_params.update({"label_week": cv_week})
        cv_score = full_cv_run(
            transactions_df, customers_df, articles_df, generate_candidates_func, model_report_func, **cv_params
        )

        cv_scores.append(cv_score)
        duration = datetime.now() - starting_time
        total_duration += duration
        print(f"Finished cv of week {cv_week} in {duration}. Score: {cv_score}\n")

    average_scores = round(np.mean(cv_scores), 5)
    print(
        f"Finished all {len(cv_weeks)} cvs in {total_duration}. "
        f"Average cv score: {average_scores}"
    )

    return average_scores


def full_sub_train_run(t, c, a, cand_features_func, score_func, **kwargs):
    (
        train_ids_df,
        train_X,
        train_group_lengths,
        train_y,
        train_truth_df,
    ) = prepare_concat_train_modeling_dfs(t, c, a, cand_features_func, **kwargs)

    model = LGBMRanker(**(kwargs["lgbm_params"]))

    eval_set = [(train_X, train_y)]
    eval_group = [train_group_lengths]
    eval_names = ["train"]

    le_callback = lightgbm.log_evaluation(kwargs["log_evaluation"])

    model.fit(
        train_X,
        train_y,
        eval_set=eval_set,
        eval_names=eval_names,
        eval_group=eval_group,
        eval_metric="MAP",
        eval_at=kwargs["eval_at"],
        callbacks=[le_callback],
        group=train_group_lengths,
    )

    # save/reload model
    model_path = f"model_{kwargs['label_week']}"
    model.booster_.save_model(model_path)
    model = lightgbm.Booster(model_file=model_path)
    # train predictions and scores
    train_pred = model.predict(train_X)
    print("Train AUC {:.4f}".format(roc_auc_score(train_y, train_pred)))
    print("Train score: ", score_func(train_ids_df, train_pred, train_truth_df))

    del train_X, train_y, train_group_lengths, train_truth_df, train_pred


def full_sub_predict_run(
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        articles_df: pd.DataFrame, 
        generate_candidates_func: callable, 
        **kwargs
    ):
    customer_batches = []
    customers = customers_df["customer_id"].unique().tolist()
    customer_batches.append(customers[: len(customers)//3])
    customer_batches.append(customers[len(customers)//3: len(customers)//3 * 2])
    customer_batches.append(customers[len(customers)//3 * 2:])

    batch_preds = []
    for idx, customer_batch in enumerate(customer_batches):
        print(
            f"Generating candidates/features for batch #{idx+1} of {len(customer_batches)}"
        )

        sub_X, _ = generate_candidates_func(
            transactions_df, 
            customers_df, 
            articles_df, 
            customer_batch=customer_batch, 
            **kwargs
        )
        sub_ids_df = sub_X[["customer_id", "article_id"]]
        sub_X = sub_X.drop(["customer_id", "article_id"], axis=1)

        print(
            f"candidate/features shape of batch: ({sub_X.shape[0]:,}, {sub_X.shape[1]})",
        )

        prediction_models = kwargs.get("prediction_models")
        model_nums = len(prediction_models)

        first_model_path = prediction_models[0]
        first_model = lightgbm.Booster(model_file=first_model_path)

        print(f"predicting with '{first_model_path}'")
        sub_pred = pred_in_batches(first_model, sub_X) / model_nums
        del first_model

        for model_path in prediction_models[1:]:
            model = lightgbm.Booster(model_file=model_path)
            print(f"predicting with '{model_path}'")
            sub_pred2 = pred_in_batches(model, sub_X)
            del model
            sub_pred += sub_pred2 / model_nums

        batch_preds.append(create_predictions(sub_ids_df, sub_pred))

        del sub_ids_df, sub_X, sub_pred

    predictions = pd.concat(batch_preds)

    return predictions