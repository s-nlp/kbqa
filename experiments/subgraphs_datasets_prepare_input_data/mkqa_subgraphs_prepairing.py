import pandas as pd
from pywikidata import Entity
from tqdm.auto import tqdm
import ujson
import datasets
from wd_api import get_wd_search_results
from multiprocessing import Pool, cpu_count

model_name = 't5-large-ssm'
predictions_path = f'../../{model_name}-res/google_{model_name}/evaluation/version_0/results.csv'


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
    try:
        elastic_results = get_wd_search_results(label, top_k, language='en')[:top_k]
    except:
        elastic_results = []

    try:
        elastic_results.extend(
            get_wd_search_results(label.replace("\"", "").replace("\'", "").strip(), top_k, language='en')[:top_k]
        )
    except:
        return None

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
    print(f'Start: {row['id']}')
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

    print(f'End: {row['id']}')
    return results


def eval_df(df):
    num_processes = cpu_count()
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
        ds['test'].to_pandas(),
        on=['question'],
    )

    results = eval_df(test_df)
    with open(f'../../{model_name}_test.jsonl', 'w') as f:
        for data_line in results:
            f.write(ujson.dumps(data_line) + '\n')
