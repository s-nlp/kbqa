import pandas as pd
import numpy as np
from datasets import Dataset


def merge_datasets(
    mintaka_ds: Dataset, outputs_ds: Dataset, features_ds: Dataset
) -> pd.DataFrame:
    output_df = outputs_ds.to_pandas()
    on_keys = ["question"]
    if "id" in output_df.columns:
        on_keys.append("id")
        
    outputs_df = pd.merge(
        mintaka_ds.to_pandas(),
        output_df,
        on=on_keys,
        how="left",
    )
    

    outputs_columns = list(outputs_ds.features.keys())
    if "id" not in output_df.columns:
        outputs_columns = ["id"] + outputs_columns

    merged_df = pd.merge(
        outputs_df[outputs_columns],
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



def str_to_arr(strng):
    """str to arr"""
    arr = strng.split(",")
    arr = [float(a) for a in arr]
    return np.array(arr)



def prepare_data(
    mintaka_ds: Dataset, outputs_ds: Dataset, features_ds: Dataset
) -> pd.DataFrame:
    df = merge_datasets(mintaka_ds, outputs_ds, features_ds)
    df = compile_seq2seq_outputs_to_model_answers_column(df)
    for column in [
        'determ_sequence_embedding',
        'gap_sequence_embedding',
        't5_sequence_embedding',
        'question_answer_embedding',

    ]:
        df[column] = df[column].apply(
            lambda embedding: embedding if isinstance(embedding, (list, np.ndarray, float)) else str_to_arr(embedding)
        )
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
