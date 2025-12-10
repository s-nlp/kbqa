"""Prompt generation and parsing utilities for RankGPT"""

import re
from typing import List, Dict, Any, Optional


def create_ranking_prompt(question: str, answers: List[Dict[str, Any]], window_start: int = 1) -> str:
    """
    Generate the instructional permutation prompt for RankGPT.
    
    Args:
        question: The question to rank answers for
        answers: List of answer dictionaries with 'answer_string' and optionally 'entity_id' and 'graph_sequence'
        window_start: Starting number for answer indexing (default 1)
        
    Returns:
        Formatted prompt string
    """
    num_answers = len(answers)
    
    # Check if any answer has a graph sequence
    has_graph_sequences = any("graph_sequence" in answer and answer.get("graph_sequence") for answer in answers)
    
    # Create the main prompt
    if has_graph_sequences:
        prompt = f"""I will provide you with a question and {num_answers} candidate answers. Each answer is associated with a knowledge graph subgraph sequence that shows the reasoning path. Rank the answers by their relevance to the question, considering both the answer text and the graph structure that supports it. The most relevant answer should be ranked first, and the least relevant answer should be ranked last.

Question: {question}

Candidate Answers with Graph Context:
"""
    else:
        prompt = f"""I will provide you with a question and {num_answers} candidate answers. Rank the answers by their relevance to the question, from most relevant to least relevant.

Question: {question}

Candidate Answers:
"""
    
    # Add numbered answer candidates with graph sequences if available
    for i, answer in enumerate(answers, start=window_start):
        answer_text = answer["answer_string"]
        graph_sequence = answer.get("graph_sequence")
        
        if graph_sequence and has_graph_sequences:
            prompt += f"[{i}] Answer: {answer_text}\n"
            prompt += f"    Graph Sequence: {graph_sequence}\n"
        else:
            prompt += f"[{i}] {answer_text}\n"
    
    prompt += f"""
Please rank the {num_answers} answers above. The most relevant answer should be ranked first, and the least relevant answer should be ranked last. Please give the ranking results in the format [x] > [y] > [z] > ... where x, y, z are the numbers of the answers in order of relevance. Don't include any other text in your response, your response will be parsed automatically.

Ranking:"""
    
    return prompt


def parse_ranking_output(llm_response: str, num_answers: int) -> Optional[List[int]]:
    """
    Parse LLM response to extract ranking.
    
    Args:
        llm_response: Raw response from the LLM
        num_answers: Number of answers that were ranked
        
    Returns:
        List of indices in ranked order, or None if parsing failed
    """
    if not llm_response:
        return None
    
    # Clean the response
    response = llm_response.strip()
    
    # Try to find ranking pattern like [1] > [2] > [3] or [3] > [1] > [2]
    ranking_patterns = [
        r'\[(\d+)\]\s*>\s*\[(\d+)\](?:\s*>\s*\[(\d+)\])*',  # [1] > [2] > [3] format
        r'(\d+)\s*>\s*(\d+)(?:\s*>\s*(\d+))*',  # 1 > 2 > 3 format
    ]
    
    for pattern in ranking_patterns:
        matches = re.findall(pattern, response)
        if matches:
            # Extract all numbers from the first match
            numbers = []
            for match in matches[0]:
                if match:
                    numbers.append(int(match))
            
            # Validate that we have the right number of answers
            if len(numbers) == num_answers and all(1 <= num <= num_answers for num in numbers):
                return numbers
    
    # Fallback: try to extract any sequence of numbers
    number_pattern = r'\b(\d+)\b'
    numbers = re.findall(number_pattern, response)
    if numbers:
        numbers = [int(n) for n in numbers]
        # Filter to valid answer indices
        valid_numbers = [n for n in numbers if 1 <= n <= num_answers]
        if len(valid_numbers) == num_answers:
            return valid_numbers
    
    return None


def create_sliding_window_prompt(question: str, answers: List[Dict[str, Any]], 
                               window_start: int, window_end: int) -> str:
    """
    Create prompt for a sliding window of answers.
    
    Args:
        question: The question to rank answers for
        answers: List of answer dictionaries
        window_start: Starting index (1-based)
        window_end: Ending index (1-based, inclusive)
        
    Returns:
        Formatted prompt string for the window
    """
    window_answers = answers[window_start-1:window_end]
    return create_ranking_prompt(question, window_answers, window_start)


def extract_ranking_from_response(response: str, expected_count: int) -> Optional[List[int]]:
    """
    Extract ranking from LLM response with better error handling.
    
    Args:
        response: LLM response text
        expected_count: Expected number of ranked items
        
    Returns:
        List of ranked indices or None if parsing failed
    """
    if not response:
        return None
    
    # Try the main parsing function first
    ranking = parse_ranking_output(response, expected_count)
    if ranking:
        return ranking
    
    # Fallback strategies
    fallback_patterns = [
        r'(\d+)(?:\s*,\s*(\d+))*',  # 1, 2, 3 format
        r'(\d+)(?:\s+(\d+))*',  # 1 2 3 format
    ]
    
    for pattern in fallback_patterns:
        matches = re.findall(pattern, response)
        if matches:
            numbers = []
            for match in matches[0]:
                if match:
                    numbers.append(int(match))
            
            if len(numbers) == expected_count and all(1 <= num <= expected_count for num in numbers):
                return numbers
    
    return None
