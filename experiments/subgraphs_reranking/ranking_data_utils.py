""" utils for ranking answers """
import pandas as pd
import numpy as np
from datasets import Dataset


def merge_datasets(
    mintaka_ds: Dataset, outputs_ds: Dataset, features_ds: Dataset
) -> pd.DataFrame:
    """merge mintaka, vanilla LLM outputs and subgraph datasets"""
    mintaka_df =mintaka_ds.to_pandas()
    outputs_df = pd.merge(
        mintaka_df[mintaka_df['lang'] == 'en'],
        outputs_ds.to_pandas(),
        on="question",
        how="left",
    )
    if "id_x" in outputs_df.columns:
        outputs_df.rename(columns={"id_x": "id"}, inplace=True)

    merged_df = pd.merge(
        outputs_df[["id"] + list(outputs_ds.features.keys())],
        features_ds.to_pandas(),
        on=["question"],
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
    
    if "id_x" in dataframe.columns:
        dataframe.rename(columns={"id_x": "id"}, inplace=True)
    dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()]

    return dataframe


def parse_embedding_string(embedding_str):
    """Parse comma-separated embedding string to numpy array, replacing NaN/Inf with 0.0"""
    if isinstance(embedding_str, (list, np.ndarray)):
        arr = np.array(embedding_str, dtype=np.float32)
    elif isinstance(embedding_str, str):
        try:
            arr = np.array([float(x) for x in embedding_str.split(",")], dtype=np.float32)
        except (ValueError, AttributeError):
            arr = np.array([0.0], dtype=np.float32)
    else:
        arr = np.array([0.0], dtype=np.float32)
    
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def convert_embedding_columns_to_arrays(dataframe: pd.DataFrame, embedding_columns: list) -> pd.DataFrame:
    """Convert embedding string columns to numpy arrays, handling NaN/Inf values"""
    dataframe = dataframe.copy()
    for col in embedding_columns:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].apply(parse_embedding_string)
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
