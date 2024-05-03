""" utils for ranking answers """
import pandas as pd
import numpy as np
from datasets import Dataset


def merge_datasets(
    mintaka_ds: Dataset, outputs_ds: Dataset, features_ds: Dataset
) -> pd.DataFrame:
    """merge mintaka, vanilla LLM outputs and subgraph datasets"""
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


def compile_seq2seq_outputs_to_model_answers_column(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """return a column of the vanilla LLM outputs"""
    answers_columns = [c for c in dataframe.columns if c.startswith("answer_")]
    dataframe["model_answers"] = dataframe[answers_columns].values.tolist()
    dataframe.drop(answers_columns, axis=1, inplace=True)
    return dataframe


def prepare_data(
    mintaka_ds: Dataset, outputs_ds: Dataset, features_ds: Dataset
) -> pd.DataFrame:
    """merge mintaka, vanilla LLM outputs and subgraph datasets"""
    dataframe = merge_datasets(mintaka_ds, outputs_ds, features_ds)
    dataframe = compile_seq2seq_outputs_to_model_answers_column(dataframe)
    return dataframe


def df_to_features_array(dataframe: pd.DataFrame) -> np.ndarray:
    """convert from df to arr representation"""
    features_array = []
    for column in dataframe.columns:
        # If value in this column a list or ndarray, then this column contains embeddings
        is_embedding_column = isinstance(dataframe[column].iloc[0], (list, np.ndarray))

        if is_embedding_column:
            features_array.append(np.vstack(dataframe[column].values))
        else:
            features_array.append(np.expand_dims(dataframe[column].values, axis=1))
    return np.hstack(features_array)
