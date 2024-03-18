"""train catboost regressor for reranking"""
import argparse
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sklearn import preprocessing
from catboost import Pool, CatBoostRegressor

parse = argparse.ArgumentParser()
parse.add_argument(
    "run_name",
    type=str,
    help="folder name inside output_path for storing model. Also used for wandb",
)
parse.add_argument(
    "--features_data_path",
    type=str,
    default="s-nlp/Mintaka_Graph_Features_T5-xl-ssm",
    help="Path to train sequence data file (HF)",
)
parse.add_argument(
    "--with_tfidf",
    type=bool,
    default=False,
    help="to whether or not use TDA tfidf to train",
)
parse.add_argument(
    "--num_iters",
    type=int,
    default=100000,
    help="number of iterations for catboost",
)
parse.add_argument(
    "--early_stopping_rounds",
    type=int,
    default=1000,
    help="number of iterations for catboost",
)
parse.add_argument(
    "--save_folder_path",
    type=str,
    default="/workspace/storage/misc/features_reranking",
    help="path for the results folder ",
)
parse.add_argument(
    "--sequence_type",
    type=str,
    default="g2t",
    choices=["g2t", "determ", "gap", "all"],
    help="path for the results folder ",
)


def filter_df_sequence(dataframe, seq_type):
    """filter df base on the sequence type,
    return filtered df & textual + embedding features"""

    textual_feat = ["determ_sequence", "g2t_sequence", "gap_sequence"]
    if seq_type != "all":
        if seq_type == "g2t":
            rm_lst = ["determ_sequence", "gap_sequence"]
        elif seq_type == "determ":
            rm_lst = ["g2t_sequence", "gap_sequence"]
        elif seq_type == "gap":
            rm_lst = ["determ_sequence", "g2t_sequence"]
        drop_text_cols = [item for item in textual_feat if item in rm_lst]
        drop_em_cols = [f"{feat}_embedding" for feat in drop_text_cols]
        dataframe = dataframe.drop(drop_text_cols + drop_em_cols, axis=1)
        textual_feat = [f"{seq_type}_sequence"]
        emb_feat = [f"{seq_type}_sequence_embedding"]
    else:
        emb_feat = [f"{feat}_embedding" for feat in textual_feat]

    return dataframe, textual_feat, emb_feat


def str_to_arr(strng):
    """str to arr"""
    arr = strng.split(",")
    arr = [float(a) for a in arr]
    return np.array(arr)


def apply_col_scale(dataframe, col, scaler):
    """apply min max scaling"""
    dataframe[col] = scaler.fit_transform(dataframe[col])
    return dataframe


def get_numeric_cols(dataframe):
    """return all cols with numeric features"""
    cols_numeric = []
    for col_name, col_type in dataframe.dtypes.to_dict().items():
        if (
            col_type is np.dtype("int64") or col_type is np.dtype("float64")
        ) and col_name != "correct":
            cols_numeric.append(col_name)

    return cols_numeric


def process_text_features(train, test, text_feat):
    """process all text features"""
    for feat in text_feat:
        train[feat] = train[feat].astype("string")
        train[feat] = train[feat].astype("string")
    return train, test


def process_numeric_features(train, test):
    """process all numerical features (scaling)"""
    min_max_scaler = preprocessing.MinMaxScaler()
    train_numeric_cols = get_numeric_cols(train)
    test_numeric_cols = get_numeric_cols(test)

    train = apply_col_scale(train, train_numeric_cols, min_max_scaler)
    test = apply_col_scale(test, test_numeric_cols, min_max_scaler)
    return train, test


def process_embedding_features(train, test, embedding_features, with_tdidf):
    """proccess all embedding features"""
    if not with_tdidf:
        train = train.drop("tfidf_vector", axis=1)
        test = test.drop("tfidf_vector", axis=1)
    else:  # with tfidf
        embedding_features.append("tfidf_vector")

    for curr_feat in embedding_features:
        train[curr_feat] = train[curr_feat].apply(str_to_arr)
        test[curr_feat] = test[curr_feat].apply(str_to_arr)
    return train, test, embedding_features


def find_weight(target):
    """find weight for imbalanced classification"""
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)]
    )
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in target])

    samples_weight = np.double(samples_weight)
    return samples_weight


if __name__ == "__main__":
    args = parse.parse_args()
    save_path = Path(args.save_folder_path) / Path(args.run_name)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    graph_features_ds = load_dataset(args.features_data_path)
    train_df = graph_features_ds["train"].to_pandas()
    test_df = graph_features_ds["test"].to_pandas()

    # process numeric features
    train_df, test_df = process_numeric_features(train_df, test_df)

    # filter dataset based on which seq type and process textual features
    text_features = ["question_answer"]
    emb_features = ["question_answer_embedding"]
    train_df, textual_feats, embd_feats = filter_df_sequence(
        train_df, args.sequence_type
    )
    test_df, _, _ = filter_df_sequence(test_df, args.sequence_type)
    text_features += textual_feats
    emb_features += embd_feats
    train_df, test_df = process_text_features(train_df, test_df, text_features)

    # process the embedding features
    train_df, test_df, emb_features = process_embedding_features(
        train_df, test_df, emb_features, args.with_tfidf
    )

    X_train = train_df.drop(["correct", "question"], axis=1)
    X_test = test_df.drop(["correct", "question"], axis=1)
    y_train = train_df["correct"].tolist()
    y_test = test_df["correct"].tolist()

    learn_pool = Pool(
        X_train,
        y_train,
        text_features=text_features,
        feature_names=list(X_train),
        embedding_features=emb_features,
        weight=find_weight(y_train),
    )

    test_pool = Pool(
        X_test,
        y_test,
        text_features=text_features,
        feature_names=list(X_test),
        embedding_features=emb_features,
        weight=find_weight(y_test),
    )

    # hyper-params tuning
    params = {
        "learning_rate": [0.03, 0.1],
        "depth": [4, 6, 10],
        "l2_leaf_reg": [5, 7, 9, 11],
    }
    model = CatBoostRegressor()
    grid_search_result = model.grid_search(params, learn_pool)

    model = CatBoostRegressor(
        iterations=args.num_iters,
        learning_rate=grid_search_result["params"]["learning_rate"],
        depth=grid_search_result["params"]["depth"],
        l2_leaf_reg=grid_search_result["params"]["l2_leaf_reg"],
        # task_type="GPU",
        early_stopping_rounds=args.early_stopping_rounds,
        eval_metric="RMSE",
    )
    model.fit(learn_pool, eval_set=test_pool, verbose=200)
    model.save_model(save_path / "best_model")
