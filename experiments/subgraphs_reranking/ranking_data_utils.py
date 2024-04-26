import pandas as pd
import numpy as np
from datasets import Dataset


def merge_datasets(
    mintaka_ds: Dataset, outputs_ds: Dataset, features_ds: Dataset
) -> pd.DataFrame:
    outputs_df = pd.merge(
        mintaka_ds.to_pandas(),
        outputs_ds.to_pandas(),
        on="question",
        how="left",
    )
    merged_df = pd.merge(
        outputs_df[["id"] + list(outputs_ds.features.keys())],
        features_ds.to_pandas(),
        on=["id", "question"],
        how="left",
    )
    return merged_df


def compile_seq2seq_outputs_to_model_answers_column(df: pd.DataFrame) -> pd.DataFrame:
    answers_columns = [c for c in df.columns if c.startswith("answer_")]
    df["model_answers"] = df[answers_columns].values.tolist()
    df.drop(answers_columns, axis=1, inplace=True)
    return df


def prepare_data(
    mintaka_ds: Dataset, outputs_ds: Dataset, features_ds: Dataset
) -> pd.DataFrame:
    df = merge_datasets(mintaka_ds, outputs_ds, features_ds)
    df = compile_seq2seq_outputs_to_model_answers_column(df)
    return df

def df_to_features_array(df: pd.DataFrame) -> np.ndarray:
    """convert from df to arr representation"""
    features_array = []
    for column in df.columns:
        # If value in this column a list or ndarray, then this column contains embeddings
        is_embedding_column = isinstance(df[column].iloc[0], (list, np.ndarray))

        if is_embedding_column:
            features_array.append(np.vstack(df[column].values))
        else:
            features_array.append(np.expand_dims(df[column].values, axis=1))
    return np.hstack(features_array)
