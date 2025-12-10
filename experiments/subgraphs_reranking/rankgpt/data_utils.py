"""Data preparation utilities for RankGPT ranking"""

import pandas as pd
from typing import List, Dict, Any, Optional


def prepare_rankgpt_data(test_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert test DataFrame to RankGPT input format.
    
    Args:
        test_df: DataFrame with columns ['id', 'question', 'model_answers', 'answerEntity']
        
    Returns:
        List of dictionaries with question_id, question, and unique_answers
    """
    rankgpt_data = []
    
    for question_id, group in test_df.groupby("id"):
        question = group["question"].iloc[0]
        
        # Extract unique answers - deduplicate based on answer string or entity ID
        unique_answers = []
        seen_answers = set()
        
        # Get model answers (list of answer strings)
        model_answers = group["model_answers"].iloc[0]
        
        # Get answer entities if available
        answer_entities = group["answerEntity"].tolist() if "answerEntity" in group.columns else [None] * len(model_answers)
        
        for i, (answer_str, entity_id) in enumerate(zip(model_answers, answer_entities)):
            # Create unique identifier for deduplication
            if entity_id and entity_id != "None":
                unique_key = f"entity:{entity_id}"
            else:
                unique_key = f"string:{answer_str}"
            
            if unique_key not in seen_answers:
                seen_answers.add(unique_key)
                unique_answers.append({
                    "index": i,
                    "answer_string": answer_str,
                    "entity_id": entity_id if entity_id and entity_id != "None" else None
                })
        
        rankgpt_data.append({
            "question_id": question_id,
            "question": question,
            "unique_answers": unique_answers
        })
    
    return rankgpt_data


def extract_unique_answers_from_group(group: pd.DataFrame, graph_sequence_feature: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract unique answers from a single question group.
    
    Args:
        group: DataFrame group for a single question
        graph_sequence_feature: Optional feature name for graph sequence (e.g., 'highlighted_determ_sequence')
        
    Returns:
        List of unique answer dictionaries with optional graph sequences
    """
    unique_answers = []
    seen_answers = set()
    
    # Get model answers (list of answer strings) - this is the source of truth for what to rank
    model_answers = group["model_answers"].iloc[0]
    
    # Iterate through group rows - each row represents one question/answer candidate pair
    for row_idx, (idx, row) in enumerate(group.iterrows()):
        # Get answer entity from this row
        entity_id = row.get("answerEntity") if "answerEntity" in row else None
        if entity_id and pd.notna(entity_id):
            entity_id = str(entity_id)
            if entity_id == "None":
                entity_id = None
        else:
            entity_id = None
        
        # Get answer string from this row
        # Try question_answer column first, then fallback to model_answers by index
        answer_str = None
        if "question_answer" in row and pd.notna(row.get("question_answer")):
            qa = str(row["question_answer"])
            if ";" in qa:
                answer_str = qa.split(";")[-1].strip()
        elif row_idx < len(model_answers):
            answer_str = model_answers[row_idx]
        
        if not answer_str:
            continue
        
        # Create unique identifier for deduplication
        if entity_id and entity_id != "None":
            unique_key = f"entity:{entity_id}"
        else:
            unique_key = f"string:{answer_str}"
        
        if unique_key not in seen_answers:
            seen_answers.add(unique_key)
            
            # Get graph sequence directly from this row if feature is specified
            graph_sequence = None
            if graph_sequence_feature and graph_sequence_feature in row:
                graph_sequence = row.get(graph_sequence_feature)
                if graph_sequence and pd.notna(graph_sequence):
                    graph_sequence = str(graph_sequence)
                else:
                    graph_sequence = None
            
            answer_dict = {
                "index": row_idx,
                "answer_string": answer_str,
                "entity_id": entity_id
            }
            
            if graph_sequence:
                answer_dict["graph_sequence"] = graph_sequence
            
            unique_answers.append(answer_dict)
    
    return unique_answers
