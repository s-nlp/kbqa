import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
import argparse

from ast import literal_eval
from calculate_metric import calculate_metric
from catboost import Pool, CatBoostRegressor
from pathlib import Path
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


parse = argparse.ArgumentParser()

parse.add_argument(
    "--save_path",
    type=str,
    default='catboost_models/mistral_all',
)

parse.add_argument(
    "--candidate_path",
    type=str,
    default='../../kbqa/generated_candidates/finetune_mistral_3_50.pkl',
)

parse.add_argument(
    "--sequence_type",
    type=str,
    choices=["g2t", "determ", "gap", "all"],
    default='all',  
)

parse.add_argument(
    "--finetune",
    type=bool,
    default=True,  
)

parse.add_argument(
    "--with_tdidf",
    type=bool,
    default=False,  
)

parse.add_argument(
    "--use_catboost",
    type=bool,
    default=True,  
)


def try_literal_eval(s):
    try:
        return literal_eval(s)
    except ValueError:
        return s

def arr_to_str(arr):
    arr = list(arr)
    return ",".join(str(a) for a in arr)

def str_to_arr(string):
    arr = string.split(",")
    arr = [float(a) for a in arr]
    return np.array(arr)

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
    
def process_embedding_features(dataframe, emb_features):
    """process all embedding features in dataframe"""
    dataframe = add_processed_emb(dataframe, emb_features)
    return dataframe

def process_text_features(df, text_features):
    for text_feat in text_features:
        df[text_feat] = df[text_feat].astype("string")
    return df

def get_numeric_cols(df):
    """return all cols with numeric features"""
    cols_numeric = []
    for k, v in df.dtypes.to_dict().items():
        if (v is np.dtype("int64") or v is np.dtype("float64")) and k != "correct":
            cols_numeric.append(k)
    return cols_numeric

def apply_col_scale(df, col, scale_type, min_max_scaler):
    """apply min max scaling"""
    if scale_type == 'train':
        df[col] = min_max_scaler.fit_transform(df[col])
    else:
        df[col] = min_max_scaler.transform(df[col])
    return df

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


if __name__ == "__main__":
    args = parse.parse_args()
    
    processed_train_df = pd.read_csv('new_mistral_train.csv')
    processed_val_df = pd.read_csv('new_mistral_validation.csv')
    processed_test_df = pd.read_csv('new_mistral_test.csv')

    save_path = Path(args.save_path)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(args.candidate_path, 'rb') as file:
        model_answers = pickle.load(file)

    # get the numerica cols in train and test to scale
    train_numeric_cols = get_numeric_cols(processed_train_df)
    val_numeric_cols = get_numeric_cols(processed_val_df)
    test_numeric_cols = get_numeric_cols(processed_test_df)

    min_max_scaler = preprocessing.MinMaxScaler()
    processed_train_df = apply_col_scale(processed_train_df, train_numeric_cols, 'train', min_max_scaler)
    processed_val_df = apply_col_scale(processed_val_df, val_numeric_cols, 'val', min_max_scaler)
    processed_test_df = apply_col_scale(processed_test_df, test_numeric_cols, 'test', min_max_scaler)

    text_features = ["question_answer"]
    emb_features = ["question_answer_embedding"]
    processed_train_df, textual_feats, embd_feats = filter_df_sequence(
        processed_train_df, args.sequence_type
    )
    processed_val_df, _, _ = filter_df_sequence(processed_val_df, args.sequence_type)
    processed_test_df, _, _ = filter_df_sequence(processed_test_df, args.sequence_type)
    text_features += textual_feats
    emb_features += embd_feats

    processed_train_df = process_text_features(processed_train_df, text_features)
    processed_val_df = process_text_features(processed_val_df, text_features)
    processed_test_df = process_text_features(processed_test_df, text_features)

    processed_train_df = processed_train_df.dropna()
    processed_val_df = processed_val_df.dropna()
    processed_test_df = processed_test_df.dropna()

    if not args.with_tdidf:
        processed_train_df = processed_train_df.drop("tfidf_vector", axis=1)
        processed_val_df = processed_val_df.drop("tfidf_vector", axis=1)
        processed_test_df = processed_test_df.drop("tfidf_vector", axis=1)
    else:  # with tfidf
        emb_features.append("tfidf_vector")
        
    for e_f in emb_features:
        processed_train_df[e_f] = processed_train_df[e_f].apply(str_to_arr)
        processed_val_df[e_f] = processed_val_df[e_f].apply(str_to_arr)
        processed_test_df[e_f] = processed_test_df[e_f].apply(str_to_arr)

    if args.use_catboost: 
        
        drop_cols = ["correct", "question"]
        X_train = processed_train_df.drop(drop_cols, axis=1)
        y_train = processed_train_df["correct"].tolist()
        X_test = processed_val_df.drop(drop_cols, axis=1)
        y_test = processed_val_df["correct"].tolist()

        learn_pool = Pool(
            X_train,
            y_train,
            text_features=text_features,
            feature_names=list(X_train),
            embedding_features=emb_features,
        )

        test_pool = Pool(
            X_test,
            y_test,
            text_features=text_features,
            feature_names=list(X_test),
            embedding_features=emb_features,
        )

        if args.finetune:
            print("finetuning")
            params = {
                "learning_rate": list(np.linspace(0.03, 0.3, 5)),
                "depth": [4, 6, 8, 10],
                "iterations": [2000, 3000, 4000],
            }
            model = CatBoostRegressor(task_type="GPU", verbose=500)
            grid_search_result = model.grid_search(params, learn_pool)

            lr = grid_search_result["params"]["learning_rate"]
            depth = grid_search_result["params"]["depth"]
            iterations = grid_search_result["params"]["iterations"]
        else:
            lr = 0.2
            depth = 6
            iterations = 2000

        print('iterations:', iterations)
        print('lr', lr)
        print('depth:', depth)
            
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=lr,
            depth=depth,
            task_type="GPU",
            early_stopping_rounds=1000,
            eval_metric="RMSE",
            verbose=200
        )

        model.fit(learn_pool, eval_set=test_pool)
        model.save_model(save_path / "best_model")
        
    else:
        processed_train_df = process_embedding_features(processed_train_df, emb_features)
        processed_test_df = process_embedding_features(processed_test_df, emb_features)
        processed_val_df = process_embedding_features(processed_val_df, emb_features)
        
        drop_cols = ["correct", "question", "question_answer"] + text_features + emb_features
        X_train = processed_train_df.drop(drop_cols, axis=1)
        y_train = processed_train_df["correct"].tolist()
        X_test = processed_val_df.drop(drop_cols, axis=1)
        y_test = processed_val_df["correct"].tolist() 
        
        errors = []
        best_alpha = 1
        for alpha in tqdm(np.logspace(-2, 2, 20)):
            regr = linear_model.LinearRegression(C=alpha)
            regr.fit(X_train, y_train)
            y_pred = regr.predict(X_test)
            error = mean_squared_error(y_test, y_pred)
            errors.append(error)
            if min(errors) == error:
                best_alpha = alpha
        print('best_alpha:', best_alpha)
        
        model = linear_model.LinearRegression(C=best_alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
        with open(save_path / 'best_model.pkl', "wb+") as f:
            pickle.dump(model, f) 

    final_acc, hit_3200, hit_4000 = calculate_metric(model, processed_test_df, model_answers, drop_cols)
    shape_reranking = len(processed_test_df['question'].value_counts().keys().tolist())

    print('only reranking questions', final_acc)
    print('shape of reranking questions', len(shape_reranking))
    print('part', final_acc / len(shape_reranking))
    print('reranking 3200', (hit_3200 + final_acc) / 3200)
    print('reranking 4000', (hit_4000 + final_acc) / 4000)
