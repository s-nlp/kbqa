"""train catboost regressor for reranking"""
import argparse
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

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
    "--save_folder_path",
    type=str,
    default="/workspace/storage/misc/features_reranking",
    help="path for the results folder ",
)
parse.add_argument(
    "--with_tfidf",
    type=bool,
    default=True,
    help="to whether or not use TDA tfidf to train",
)
parse.add_argument(
    "--use_embeddings",
    type=bool,
    default=True,
    help="to whether use the embedding features or not",
)
parse.add_argument(
    "--sequence_type",
    type=str,
    default="g2t",
    choices=["g2t", "determ", "gap", "all"],
    help="path for the results folder ",
)


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


def process_numeric_features(train, test):
    """process all numerical features (scaling)"""
    min_max_scaler = preprocessing.MinMaxScaler()
    train_numeric_cols = get_numeric_cols(train)
    test_numeric_cols = get_numeric_cols(test)

    train = apply_col_scale(train, train_numeric_cols, min_max_scaler)
    test = apply_col_scale(test, test_numeric_cols, min_max_scaler)
    return train, test


def process_embedding_features(dataframe, emb_features):
    """process all embedding features in dataframe"""
    for em_feat in tqdm(emb_features):
        dataframe[em_feat] = dataframe[em_feat].apply(str_to_arr)

    dataframe = add_processed_emb(dataframe, emb_features)
    return dataframe


def process_embedding_feature(dataframe, em_type):
    """process one embedding of dataframe,
    split embedding into individual rows"""
    embeddings = dataframe[em_type].tolist()
    emb_dict = {}
    for emb in embeddings:
        for i, val in enumerate(emb):
            curr_key = f"{em_type}_{i}"
            if curr_key not in emb_dict:
                emb_dict[f"{em_type}_{i}"] = [val]
            else:
                emb_dict[f"{em_type}_{i}"].append(val)

    return pd.DataFrame(emb_dict)


def add_processed_emb(dataframe, emb_features):
    """add all processed embeddings to df"""
    em_df_list = []
    for em_feat in emb_features:
        em_df_list.append(process_embedding_feature(dataframe, em_feat))

    dataframe = pd.concat([dataframe] + em_df_list, axis=1)
    return dataframe


if __name__ == "__main__":
    args = parse.parse_args()
    save_path = Path(args.save_folder_path) / Path(args.run_name)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    if args.with_tfidf and not args.use_embeddings:
        raise ValueError("cannot use TFIDF embeddings if use_embeddings is False.")

    graph_features_ds = load_dataset(args.features_data_path)
    train_df = graph_features_ds["train"].to_pandas()
    test_df = graph_features_ds["test"].to_pandas()

    # prcocess numeric features
    train_df, test_df = process_numeric_features(train_df, test_df)

    # process embedding & textual features
    textual_features = [
        "determ_sequence",
        "gap_sequence",
        "g2t_sequence",
    ]
    embedding_features = [f"{text_feat}_embedding" for text_feat in textual_features]
    if args.sequence_type != "all":
        if args.sequence_type == "g2t":
            rm_lst = ["determ_sequence_embedding", "gap_sequence_embedding"]
        elif args.sequence_type == "determ":
            rm_lst = ["g2t_sequence_embedding", "gap_sequence_embedding"]
        else:
            rm_lst = ["determ_sequence_embedding", "g2t_sequence_embedding"]

        DROP_EM_FEAT = [item for item in embedding_features if item in rm_lst]
        embedding_features.remove(DROP_EM_FEAT)

    drop_cols = (
        textual_features + DROP_EM_FEAT + ["question_answer", "correct", "question"]
    )
    if args.with_tfidf:
        embedding_features += ["tfidf_vector"]
    else:
        drop_cols.append("tfidf_vector")

    train_df = process_embedding_features(train_df, embedding_features)
    test_df = process_embedding_features(test_df, embedding_features)
    print("finished processing embeddings for train and test")

    drop_cols += embedding_features  # after processing em, drop the arrays version
    X_train = train_df.drop(drop_cols, axis=1)
    X_test = test_df.drop(drop_cols, axis=1)
    y_train = train_df["correct"].tolist()
    y_test = test_df["correct"].tolist()
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
    with open(Path(save_path) / "finalized_model.sav", "wb+") as f:
        pickle.dump(regr, f)
