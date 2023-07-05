"""
Script to prepare the lcquad2.0 dataset for seq2seq
"""
import argparse
import pandas as pd
import json
import requests
from tqdm import tqdm
from joblib import Memory

memory = Memory("/tmp/cache", verbose=0)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--json_file_path",
    default="/workspace/kbqa/lcquad2.0/lcquad2_train_qald.json",
    type=str,
)

parser.add_argument(
    "--save_file_path",
    default="lcquad2_train.csv",
    type=str,
)


# Function to query Wikidata and retrieve the label for an entity ID
@memory.cache
def get_entity_label(entity_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    response = requests.get(url)
    data = response.json()
    try:
        label = data["entities"][entity_id]["labels"]["en"]["value"]
        return label
    except KeyError:
        return None


if __name__ == "__main__":
    args = parser.parse_args()

# Load the JSON data from the file
with open(args.json_file_path) as json_file:
    data = json.load(json_file)

    # Extract the questions from the loaded data
    questions = data["questions"]

    # Create a list to store the parsed data
    parsed_data = []

    # Parse each question and its corresponding fields using tqdm to track progress
    for question in tqdm(questions):
        question_id = question["id"]
        answertype = question.get("answertype")
        aggregation = question.get("aggregation")
        onlydbo = question.get("onlydbo")
        hybrid = question.get("hybrid")
        question_string = question["question"][0]["string"]
        sparql_query = question["query"]["sparql"]

        # Try to extract the answer value, handle the case if it's not present
        try:
            if "boolean" in question["answers"][0]:
                ANSWER = question["answers"][0]["boolean"]
                LABEL = None
            else:
                ANSWER = question["answers"][0]["results"]["bindings"][0]["uri"][
                    "value"
                ]
                entity_id = answer.split("/")[-1]
                try:
                    LABEL = get_entity_label(entity_id)
                except:
                    LABEL = None
        except KeyError:
            ANSWER = None
            LABEL = None

        # Append the parsed data to the list
        parsed_data.append(
            {
                "Question ID": question_id,
                "Answer Type": answertype,
                "Aggregation": aggregation,
                "OnlyDBO": onlydbo,
                "Hybrid": hybrid,
                "Question": question_string,
                "SPARQL Query": sparql_query,
                "Answer": ANSWER,
                "Label": LABEL,
            }
        )

    # Create the DataFrame
    df = pd.DataFrame(parsed_data)
    df.loc[df["Answer"] is False, "Label"] = "False"
    df.loc[df["Answer"] is True, "Label"] = "True"
    df.dropna(subset=["Label"], inplace=True)

    df.to_csv(args.save_file_path, index=False)
