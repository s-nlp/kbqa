from datasets import load_dataset
from tqdm import tqdm


def calculate_metric(model, test_df, model_answers, drop_cols, num_beams=50):
    questions = test_df['question'].value_counts().keys().tolist()
    mintaka = load_dataset('AmazonScience/mintaka')['test'].to_pandas()
    filtered = mintaka[(mintaka['complexityType'] == 'yesno') | (mintaka['complexityType'] == 'count')]
    filtered = filtered['question'].tolist()
    final_acc, hit_3200, hit_4000 = 0, 0, 0

    for question in tqdm(questions):
        if question not in filtered:
            row = test_df[test_df['question'] == question]
            is_corrects = row["correct"].astype(bool).tolist()
            curr_rows = row.drop(drop_cols, axis=1)
            preds = model.predict(curr_rows)
            max_idx = preds.argmax()

            if is_corrects[max_idx]:
                final_acc += 1
        
    dataset = load_dataset('AmazonScience/mintaka')
    for i in range(len(dataset['test'])):
        if (dataset['test'][i]['question'] in questions):
            continue
        else:
            if dataset['test'][i]['answerText'] == model_answers[num_beams * i]:
                hit_4000 += 1
                if dataset['test'][i]['complexityType'] not in ['yesno', 'count']:
                    hit_3200 += 1
        
    return final_acc, hit_3200, hit_4000
