# pylint: disable=invalid-name

import warnings
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import argparse
import pandas as pd

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()


parser.add_argument(
    "--base_dataset",
    default="/workspace/kbqa/subgraph_data.csv",
    type=str,
)
parser.add_argument(
    "--meta",
    default="/workspace/kbqa/metas.csv",
    type=str,
)


# Define training fucntion
def train_classifiers(clf_dict, X_data, y_label):
    """
    training function
    returns:
    clf: classifier
    y_pred: predicted labels
    f1: f1 score
    b_score: balanced accuracy score
    cm: confusion matrix
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_label, test_size=0.33, random_state=314, shuffle=False
    )

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    clf = next(iter(clf_dict))

    if clf == "KNeighborsClassifier":

        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = round(
            f1_score(
                y_test,
                y_pred,
                average="weighted",
            ),
            2,
        )
        b_score = balanced_accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

    elif clf == "RandomForestClassifier":

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = clf.predict(X_test)
        f1 = round(
            f1_score(
                y_test,
                y_pred,
                average="weighted",
            ),
            2,
        )
        b_score = balanced_accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

    elif clf == "LogisticRegression":

        clf = LogisticRegression(penalty="l2", max_iter=5000)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = clf.predict(X_test)
        f1 = round(
            f1_score(
                y_test,
                y_pred,
                average="weighted",
            ),
            2,
        )
        b_score = balanced_accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

    else:

        clf = XGBClassifier(
            objective="binary:logistic", colsample_bytree=0.3, learning_rate=0.1
        )
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = clf.predict(X_test)
        f1 = round(
            f1_score(
                y_test,
                y_pred,
                average="weighted",
            ),
            2,
        )
        b_score = balanced_accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

    return clf, y_pred, f1, b_score, cm


if __name__ == "__main__":

    args = parser.parse_args()

    df = pd.read_csv(args.base_dataset)
    metas = pd.read_csv(args.meta)

    data = pd.merge(metas, df, left_index=True, right_index=True)
    data = data.sort_values("idx")
    data["label"] = data["candidate_id"] == data["target_id"]
    data["label"] = data["label"].astype(int)

    numeric_features = data[
        [
            "number_of_triangles",
            "number_of_nodes",
            "number_of_edges",
            "candidate_katz_centrality",
            "candidate_pagerank",
            "subgraph_largest_clique_size",
            "shortest_path_lengths",
            "mean_shortest_path_length",
            "sitelinks",
            "outcoming_links",
            "incoming_links",
            "candidate_eigenvector_centrality",
            "candidate_clustering",
            "label",
        ]
    ]
    numeric_features = numeric_features.dropna()

    y_label = numeric_features["label"]
    X_data = numeric_features.drop("label", axis=1)

    clfs_dict = {
        "KNeighborsClassifier": {"n_neighbors": range(3, 17, 2)},
        "RandomForestClassifier": {
            "n_estimators": range(1, 15, 1),
            "max_features": range(1, 16, 1),
        },
        "LogisticRegression": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "penalty": ["l2"],
            "max_iter": [5000],
        },
        "XGBClassifier": {
            "max_depth": range(2, 10, 1),
            "alpha": range(1, 10, 1),
            "n_estimators": range(60, 220, 40),
            "learning_rate": [0.1, 0.01, 0.05],
        },
    }

    for key, value in clfs_dict.items():
        parse_dict = {key: value}
        _, _, f1, b_score, _ = train_classifiers(parse_dict, X_data, y_label)
        print("F1 score for {} = {}, Balanced_Accuracy = {}".format(key, f1, b_score))
