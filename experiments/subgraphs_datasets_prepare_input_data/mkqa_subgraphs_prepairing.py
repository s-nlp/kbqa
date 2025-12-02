import time
from collections import deque
from functools import wraps
from threading import Lock
from time import sleep

import pandas as pd
from pywikidata import Entity
from tqdm.auto import tqdm
import ujson
import datasets
from wd_api import get_wd_search_results
from multiprocessing import Pool, cpu_count

model_name = 't5-xl-ssm-nq'
type = 'train'
predictions_path = f'../../{model_name}-res/google_{model_name}/evaluation/version_0_{type}/results.csv'


def rate_limit(max_calls=15, period=60):
    def decorator(func):
        # Store state in closure variables
        calls = deque()  # Store timestamps of recent calls
        lock = Lock()  # Thread safety lock

        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                current_time = time.time()

                # Remove timestamps older than the period
                while calls and calls[0] <= current_time - period:
                    calls.popleft()

                # Check if we've exceeded the rate limit
                if len(calls) >= max_calls:
                    oldest = calls[0]
                    wait_time = oldest + period - current_time
                    if wait_time > 0:
                        print(f"Rate limit, sleep {wait_time}")
                        time.sleep(wait_time)
                        # After sleeping, update current_time and clean old calls again
                        current_time = time.time()
                        while calls and calls[0] <= current_time - period:
                            calls.popleft()

                # Record this call and execute the function
                calls.append(current_time)

            return func(*args, **kwargs)
        return wrapper
    return decorator


@rate_limit(max_calls=30, period=60)
def label_to_entity(label: str, top_k: int = 3) -> list:
    """label_to_entity method to  linking label to WikiData entity ID
    by using elasticsearch Wikimedia public API
    Supported only English language (en)

    Parameters
    ----------
    label : str
        label of entity to search
    top_k : int, optional
        top K results from WikiData, by default 3

    Returns
    -------
    list[str] | None
        list of entity IDs or None if not found
    """
    retry = True
    while retry:
        try:
            elastic_results = get_wd_search_results(label, top_k, language='en')[:top_k]
        except Exception as e:
            print(f"First e: {e}")
            if '429' in str(e):
                # print(f"Retry first for: {e}")
                sleep(1001)
            else:
                retry = False
                elastic_results = []
        else:
            retry = False

    retry = True
    while retry:
        try:
            elastic_results.extend(
                get_wd_search_results(label.replace("\"", "").replace("\'", "").strip(), top_k, language='en')[:top_k]
            )
        except Exception as e:
            print(f"Second e: {e}")
            if '429' in str(e):
                sleep(1001)
            else:
                retry = False
        else:
            retry = False

    return list(dict.fromkeys(elastic_results).keys())[:top_k]


def data_to_subgraphs(df):
    for _, row in tqdm(df.iterrows(), total=df.index.size):
        # if row['complexityType'] not in ['count', 'yesno']:
        question_entity_ids = [e['name'] for e in row['questionEntity'] if e['entityType'] == 'entity']

        for candidate_label in dict.fromkeys(row['model_answers']).keys():
            for candidate_entity_id in label_to_entity(candidate_label):
                candidate_entity = Entity(candidate_entity_id)
                yield {
                    'id': row['id'],
                    'question': row['question'],
                    'generatedAnswer': [candidate_label],
                    'answerEntity': [candidate_entity.idx],
                    'answerEntityLabel': [candidate_entity.label],
                    'questionEntity': question_entity_ids,
                    'groundTruthAnswerEntity': [e['name'] for e in row['answerEntity']]
                }


def process_row(row):
    results = []
    print(f"Start: {row['id']}")
    # print("HERE!")
    question_entity_ids = [e['name'] for e in row['questionEntity'] if e['entityType'] == 'entity']
    for candidate_label in dict.fromkeys(row['model_answers']).keys():
        for candidate_entity_id in label_to_entity(candidate_label):
            candidate_entity = Entity(candidate_entity_id)
            results.append({
                'id': row['id'],
                'question': row['question'],
                'generatedAnswer': [candidate_label],
                'answerEntity': [candidate_entity.idx],
                'answerEntityLabel': [candidate_entity.label],
                'questionEntity': question_entity_ids,
                'groundTruthAnswerEntity': [e['name'] for e in row['answerEntity']]
            })

    print(f"End: {row['id']}")
    return results


def eval_df(df):
    num_processes = cpu_count()
    print("Run with processes:", num_processes)
    # Convert DataFrame to list of dictionaries for processing
    rows = df.to_dict('records')
    # print(rows)
    # rows = rows[:num_processes]

    # Create pool and process rows
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_row, rows)

    # Convert results back to DataFrame
    results = [item for sublist in results for item in sublist]
    return results


if __name__ == '__main__':
    test_predictions = pd.read_csv(predictions_path)
    ds = datasets.load_dataset("Dms12/mkqa_mintaka_format_with_question_entities")

    answer_columns = [col for col in test_predictions.columns if col.startswith('answer_')]
    test_predictions['model_answers'] = test_predictions[answer_columns].values.tolist()
    test_predictions = test_predictions.drop(columns=answer_columns)

    test_df = pd.merge(
        test_predictions,
        ds[f'{type}'].to_pandas(),
        on=['question'],
    )

    results = eval_df(test_df)
    with open(f'../../{model_name}_{type}.jsonl', 'w') as f:
        for data_line in results:
            f.write(ujson.dumps(data_line) + '\n')
