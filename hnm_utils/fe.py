from datetime import timedelta

import pandas as pd


def create_pairs(transactions_df, week_number, pairs_per_item):
    working_t_df = transactions_df[['customer_id', 'article_id', 'week_number']].query('week_number <= @week_number').copy()
    pairs_t_df = working_t_df.query('week_number == @week_number').copy()
    
    working_t_df.drop('week_number', axis=1, inplace=True)
    pairs_t_df.drop('week_number', axis=1, inplace=True)
    working_t_df.drop_duplicates(inplace=True)
    pairs_t_df.drop_duplicates(inplace=True)
    pairs_t_df.columns = ['customer_id', 'pair_article_id']

    unique_articles = working_t_df['article_id'].unique()
    batch_size = 5000
    batch_pairs_dfs = []
    for i in range(0, len(unique_articles), batch_size):
        batch_articles = unique_articles[i : i + batch_size]
        batch_t = working_t_df[working_t_df['article_id'].isin(batch_articles)]

        # get total # of customers who bought each article
        all_cust_counts = batch_t.groupby('article_id')['customer_id'].nunique()
        all_cust_counts = all_cust_counts.reset_index()
        all_cust_counts.columns = ['article_id', 'all_customer_counts']
        all_cust_counts['all_customer_counts'] -= 1  # not him himself

        # get all pairs for those articles (other articles those customers bought)
        batch_pairs_df = (
            batch_t.merge(pairs_t_df, on='customer_id')
            .query('article_id != pair_article_id')
        )

        # delete single customer articles
        c1s = (
            batch_pairs_df.groupby('article_id')[['customer_id']]
            .nunique()
            .query('customer_id > 1')
            .index
        )
        batch_pairs_df = batch_pairs_df[batch_pairs_df['article_id'].isin(c1s)]

        # get sorted counts of article-pair occurences
        batch_pairs_df = batch_pairs_df.groupby(['article_id', 'pair_article_id'])[['customer_id']].count()
        batch_pairs_df.columns = ['customer_count']
        batch_pairs_df = batch_pairs_df.reset_index()
        batch_pairs_df = batch_pairs_df.sort_values(['article_id', 'customer_count'], ascending=False)

        # get top x pairs for each article
        batch_pairs_df = batch_pairs_df.groupby('article_id').head(pairs_per_item)

        # calculate percentage statistic
        batch_pairs_df = batch_pairs_df.merge(all_cust_counts, on='article_id')
        batch_pairs_df['percent_customers'] = (
            batch_pairs_df['customer_count'] / batch_pairs_df['all_customer_counts']
        )
        batch_pairs_df.drop('all_customer_counts', axis=1, inplace=True)

        batch_pairs_dfs.append(batch_pairs_df)

    all_article_pairs_df = pd.concat(batch_pairs_dfs)

    return all_article_pairs_df


def cal_day_numbers(dates: pd.Series):
    '''
    Calculate the number of days since the earliest date in the series

    Parameters
    ---------
    - dates: pd.Series of date strings

    Returns
    -------
    - pd.Series 
        Series of day numbers since earliest date in the original series

    '''

    dates = pd.to_datetime(dates)
    days_since_earliest = (dates - dates.min()).dt.days
    converted_series = days_since_earliest.astype('int16')
    return converted_series


def cal_week_numbers(dates: pd.Series):
    '''
    Calculate the number of weeks since the earliest date in the series

    Parameters
    ---------
    - dates: pd.Series
        Series of date strings

    Returns
    -------
    - pd.Series 
        Series of week numbers since earliest date in the original series

    '''

    dates = pd.to_datetime(dates)
    days_since_earliest = (dates - dates.min() + timedelta(1)).dt.days
    weeks_since_earliest = (days_since_earliest - 1) // 7
    converted_series = weeks_since_earliest.astype('int8')
    return converted_series


def create_cust_hier_features(transactions_df, articles_df, hier_cols, features_db):
    sample_col = "t_dat"

    # create hiers
    for hier_col in hier_cols:
        # total customer counts
        total_cust_counts = transactions_df.groupby("customer_id")[sample_col].count()

        # add hierarchy column to transactions
        article_hier_lookup = articles_df.set_index("article_id")[hier_col]
        transactions_df[hier_col] = transactions_df["article_id"].map(
            article_hier_lookup
        )

        # get customer/hierarchy statistics
        cust_hier = (
            transactions_df.groupby(["customer_id", hier_col])[sample_col]
            .count()
            .reset_index()
        )
        cust_hier.columns = list(cust_hier.columns)[:-1] + ["cust_hier_counts"]
        cust_hier = cust_hier.sort_values(
            ["customer_id", "cust_hier_counts"], ascending=False
        )

        cust_hier["total_counts"] = cust_hier["customer_id"].map(total_cust_counts)

        hier_portion_column = f"cust_{hier_col}_portion"
        cust_hier[hier_portion_column] = (
            cust_hier["cust_hier_counts"] / cust_hier["total_counts"]
        )
        cust_hier = cust_hier[["customer_id", hier_col, hier_portion_column]]
        cust_hier = cust_hier.set_index(["customer_id", hier_col])
        cust_hier[hier_portion_column] = cust_hier[hier_portion_column].astype("float32")
        features_db[hier_portion_column] = (["customer_id", hier_col], cust_hier)


def create_price_features(transactions_df, features_db):
    ###################
    # article_prices
    ###################
    article_prices_df = transactions_df.groupby("article_id")[["price"]].max()
    article_prices_df.columns = ["max_price"]

    last_week = transactions_df["week_number"].max()
    last_week_t_df = transactions_df.query(f"week_number == {last_week}")
    last_week_prices = last_week_t_df.groupby("article_id")["price"].mean()
    article_prices_df["last_week_price"] = last_week_prices

    article_prices_df = article_prices_df.dropna()

    article_prices_df["last_week_price_ratio"] = (
        article_prices_df["last_week_price"] / article_prices_df["max_price"]
    )
    article_prices_df = article_prices_df.astype("float32")

    features_db["article_prices"] = (["article_id"], article_prices_df)

    ############################
    # customer price features
    ############################
    cust_prices_df = transactions_df[
        ["customer_id", "article_id", "week_number", "price"]
    ].copy()

    cust_prices_df["max_article_price"] = cust_prices_df["article_id"].map(
        cust_prices_df.groupby(["article_id"])["price"].max()
    )

    # for each purchase, the previous article/week price, and price discount
    article_week_price_df = (
        cust_prices_df.groupby(["week_number", "article_id"])["price"]
        .mean()
        .reset_index()
    )
    article_week_price_df.columns = [
        "week_number",
        "article_id",
        "article_previous_week_price",
    ]
    article_week_price_df[
        "week_number"
    ] += 1  # for the next week, the price is from the previous week
    cust_prices_df = cust_prices_df.merge(
        article_week_price_df, on=["week_number", "article_id"]
    )
    cust_prices_df["article_previous_week_price_ratio"] = (
        cust_prices_df["article_previous_week_price"]
        / cust_prices_df["max_article_price"]
    )
    cust_prices_df = cust_prices_df.groupby("customer_id")[
        [
            "max_article_price",
            "article_previous_week_price",
            "article_previous_week_price_ratio",
        ]
    ].mean()
    cust_prices_df.columns = [
        "cust_avg_max_price",
        "cust_avg_last_week_price",
        "cust_avg_last_week_price_ratio",
    ]
    cust_prices_df = cust_prices_df.dropna()
    cust_prices_df = cust_prices_df.astype("float32")

    features_db["cust_price_features"] = (["customer_id"], cust_prices_df)


def create_cust_t_features(transactions_df, a, features_db):
    ctf_df = transactions_df.groupby("customer_id")[["sales_channel_id"]].mean()
    ctf_df.columns = ["cust_sales_channel"]
    ctf_df["cust_sales_channel"] = ctf_df["cust_sales_channel"].astype("float32")
    ctf_df["cust_sales_channel"] = ctf_df["cust_sales_channel"].round(2) - 1.0

    ctf_df["cust_t_counts"] = (
        transactions_df.groupby("customer_id")["sales_channel_id"]
        .count()
        .astype("float32")
    )
    ctf_df["cust_u_t_counts"] = (
        transactions_df.groupby("customer_id")["article_id"].nunique().astype("float32")
    )

    sub_t_df = transactions_df[["customer_id", "article_id"]].copy()
    sub_t_df["index_group_name"] = sub_t_df["article_id"].map(
        a.set_index("article_id")["index_group_name"]
    )
    gender_dict = {
        "Ladieswear": 1,
        "Baby/Children": 0.5,
        "Menswear": 0,
        "Sport": 0.5,
        "Divided": 0.5,
    }
    sub_t_df["article_gender"] = (
        sub_t_df["index_group_name"].astype("str").map(gender_dict)
    )

    sub_t_df["section_name"] = sub_t_df["article_id"].map(
        a.set_index("article_id")["section_name"].astype(str)
    )
    sub_t_df.loc[sub_t_df["section_name"] == "Ladies H&M Sport", "article_gender"] = 1
    sub_t_df.loc[sub_t_df["section_name"] == "Men H&M Sport", "article_gender"] = 0
    ctf_df["cust_gender"] = sub_t_df.groupby("customer_id")["article_gender"].mean()

    features_db["cust_t_features"] = (["customer_id"], ctf_df)


def create_art_t_features(transactions_df, features_db):
    atf_df = transactions_df.groupby("article_id")[["sales_channel_id"]].mean()
    atf_df.columns = ["art_sales_channel"]
    atf_df["art_sales_channel"] = atf_df["art_sales_channel"].astype("float32")
    atf_df["art_sales_channel"] = atf_df["art_sales_channel"].round(2) - 1.0

    atf_df["art_t_counts"] = (
        transactions_df.groupby("article_id")["sales_channel_id"]
        .count()
        .astype("float32")
    )
    atf_df["art_u_t_counts"] = (
        transactions_df.groupby("article_id")["customer_id"].nunique().astype("float32")
    )

    features_db["art_t_features"] = (["article_id"], atf_df)


def create_cust_features(customers_df, features_db):
    cust_df = customers_df.set_index("customer_id")[["age"]]
    cust_df["age"] = cust_df["age"].fillna(-1).astype("int16")

    features_db["cust_features"] = (["customer_id"], cust_df)


def create_article_cust_features(transactions_df, customers_df, features_db):
    art_cust_df = transactions_df[["article_id", "customer_id"]].drop_duplicates()
    cust_age = customers_df.set_index("customer_id")["age"]
    art_cust_df["cust_age"] = art_cust_df["customer_id"].map(cust_age)
    art_cust_df = art_cust_df.groupby("article_id")[["cust_age"]].mean()
    art_cust_df.columns = ["art_cust_age"]

    art_cust_df["art_cust_age"] = art_cust_df["art_cust_age"].fillna(-1).astype("int16")

    features_db["article_customer_age"] = (["article_id"], art_cust_df)


def create_lag_features(transactions_df, articles_df, lag_days, features_db):
    last_date = transactions_df["t_dat"].max()

    article_counts_df = pd.DataFrame(index=articles_df["article_id"])
    for lag_day in lag_days:
        # column name
        col_name = f"last_{lag_day}_days_count"

        # column values
        t_df_filtered = transactions_df[
            transactions_df["t_dat"] > (last_date - lag_day)
        ]
        lag_values = t_df_filtered.groupby("article_id")["customer_id"].nunique()

        # putting them in
        article_counts_df[col_name] = lag_values
        article_counts_df[col_name] = article_counts_df[col_name].astype("float32")

    features_db["article_counts"] = (["article_id"], article_counts_df)


def create_rebuy_features(transactions_df, features_db):
    duplicate_counts = transactions_df.groupby(["customer_id", "article_id"])[
        "week_number"
    ].count()
    duplicate_counts = duplicate_counts.sort_values().reset_index()
    duplicate_counts.columns = ["customer_id", "article_id", "buy_count"]
    rebuy_ratio_df = duplicate_counts.groupby("article_id")[["buy_count"]].mean()
    rebuy_ratio_df.columns = ["rebuy_count_ratio"]
    rebuy_ratio_df["rebuy_count_ratio"] = rebuy_ratio_df["rebuy_count_ratio"].astype(
        "float32"
    )
    duplicate_counts["buy_count"] = duplicate_counts["buy_count"].replace(1, 0)
    duplicate_counts["buy_count"][duplicate_counts["buy_count"] > 1] = 1
    rebuy_ratio_df["rebuy_ratio"] = duplicate_counts.groupby("article_id")["buy_count"].mean()
    rebuy_ratio_df = rebuy_ratio_df.astype("float32")

    features_db["rebuy_features"] = (["article_id"], rebuy_ratio_df)