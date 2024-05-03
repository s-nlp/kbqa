"""train catboost regressor for reranking"""
import argparse
from pathlib import Path
import joblib
import numpy as np
from datasets import load_dataset
from sklearn import preprocessing, utils
from catboost import Pool, CatBoostRegressor
import matplotlib.pyplot as plt

parse = argparse.ArgumentParser()
parse.add_argument(
    "run_name",
    type=str,
    help="folder name inside output_path for storing model. Also used for wandb",
)
parse.add_argument(
    "--kgqa_ds_path",
    type=str,
    default="s-nlp/KGQASubgraphsRanking",
    help="Path to train sequence data file (HF)",
)
parse.add_argument(
    "--ds_type",
    type=str,
    default="t5xlssm",
    choices=["t5largessm", "t5xlssm", "mistral", "mixtral"],
    help="Path to train sequence data file (HF)",
)
parse.add_argument(
    "--hf_cache_dir",
    type=str,
    default="/workspace/storage/misc/huggingface",
    help="path for the results folder ",
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
    default="/workspace/storage/misc/features_reranking/catboost/unified_reranking",
    help="path for the results folder ",
)
parse.add_argument(
    "--use_text_features",
    type=bool,
    default=True,
    help="To whether use the text features or not",
)
parse.add_argument(
    "--use_graph_features",
    type=bool,
    default=True,
    help="To whether use the graph features or not",
)
parse.add_argument(
    "--sequence_type",
    type=str,
    default="g2t_gap",
    choices=["g2t_t5", "g2t_determ", "g2t_gap", "none"],
    help="to whether use g2t sequences (t5, determ or gap) or not (none)",
)

features_map = {
    "text": ["question_answer_embedding"],
    "graph": [
        "num_nodes",
        "num_edges",
        "density",
        "cycle",
        "bridge",
        "katz_centrality",
        "page_rank",
        "avg_ssp_length",
    ],
    "g2t_determ": ["determ_sequence_embedding"],
    "g2t_t5": ["t5_sequence_embedding"],
    "g2t_gap": ["gap_sequence_embedding"],
}


def apply_col_scale(dataframe, col, scaler, split_type):
    """apply min max scaling to the specified columns"""
    if split_type == "train":  # train -> fit_transform
        dataframe[col] = scaler.fit_transform(dataframe[col])
    else:  # val or test -> just transform
        dataframe[col] = scaler.transform(dataframe[col])
    return dataframe


def process_numeric_features(train, val, cols, save_scaler_path):
    """process all numerical features (scaling) for train & val"""
    scaler = preprocessing.MinMaxScaler()
    train = apply_col_scale(train, cols, scaler, "train")
    val = apply_col_scale(val, cols, scaler, "validation")
    joblib.dump(scaler, save_scaler_path)
    return train, val


def plot_features_importance(trained_model, val_ds, features, path):
    """plot the features importance and save to path"""
    features_importance = trained_model.get_feature_importance(data=val_ds)

    plt.barh(features, features_importance)
    plt.tight_layout()
    plt.savefig(path)


if __name__ == "__main__":
    args = parse.parse_args()
    ds_type = args.ds_type
    save_path = Path(args.save_folder_path) / Path(ds_type) / Path(args.run_name)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    graph_features_ds = load_dataset(
        args.kgqa_ds_path,
        data_dir=f"{ds_type}_subgraphs",
        cache_dir=args.hf_cache_dir,
    )
    train_df = graph_features_ds["train"].to_pandas()
    val_df = graph_features_ds["validation"].to_pandas()

    # list of all features that we will use for this approach
    all_features_used = []

    # process embedding features
    embedding_features = []
    if args.use_text_features:  # add text feature if needed
        embedding_features += features_map["text"]
        all_features_used += features_map["text"]

    if args.sequence_type != "none":  # add g2t seq feature if needed
        embedding_features += features_map[args.sequence_type]
        all_features_used += features_map[args.sequence_type]

    if args.use_graph_features:  # add & process numeric features if needed
        train_df, val_df = process_numeric_features(
            train_df,
            val_df,
            features_map["graph"],
            Path(save_path) / "fitted_scaler.bz2",
        )
        all_features_used += features_map["graph"]

    # prepare train & val data based on features we want to use
    X_train = train_df[all_features_used]
    X_test = val_df[all_features_used]
    y_train = train_df["correct"].astype(float).tolist()
    y_test = val_df["correct"].astype(float).tolist()

    # find weights for balance classification
    train_classes = np.unique(y_train)
    train_weights = utils.compute_class_weight(
        class_weight="balanced", classes=train_classes, y=y_train
    )
    train_class_weights = np.array(y_test)
    train_class_weights[train_class_weights == 0] = train_weights[0]
    train_class_weights[train_class_weights == 1] = train_weights[1]

    learn_pool = Pool(
        X_train,
        y_train,
        feature_names=list(X_train),
        embedding_features=embedding_features,
        weight=train_class_weights,
    )

    val_pool = Pool(
        X_test,
        y_test,
        feature_names=list(X_test),
        embedding_features=embedding_features,
    )

    # hyper-params tuning
    params = {
        "learning_rate": list(np.linspace(0.03, 0.3, 5)),
        "depth": [4, 6, 8, 10],
        "iterations": [2000, 3000, 4000],
    }
    model = CatBoostRegressor()
    grid_search_result = model.grid_search(params, learn_pool)

    model = CatBoostRegressor(
        iterations=grid_search_result["params"]["iterations"],
        learning_rate=grid_search_result["params"]["learning_rate"],
        depth=grid_search_result["params"]["depth"],
        task_type="GPU",
        early_stopping_rounds=args.early_stopping_rounds,
        eval_metric="RMSE",
    )
    model.fit(learn_pool, eval_set=val_pool, verbose=200)
    model.save_model(Path(save_path) / "best_model")

    # plot and save features importance
    col_names = X_test.columns.tolist()
    save_path = plt.savefig(Path(save_path) / "features_importance.png")
    plot_features_importance(model, val_pool, col_names, save_path)
