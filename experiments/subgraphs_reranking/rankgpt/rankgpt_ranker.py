"""RankGPT ranker implementation for KGQA subgraph ranking"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ranking_model import Ranker, RankedAnswer, RankedAnswersDict
from data_utils import extract_unique_answers_from_group
from prompt_builder import create_ranking_prompt, parse_ranking_output


class RankGPTRanker(Ranker):
    """RankGPT ranker using OpenAI-compatible API for question-answer ranking"""
    
    def __init__(self, api_base: Optional[str] = None, api_key: Optional[str] = None, 
                 model_name: str = "gpt-3.5-turbo", window_size: int = 20, 
                 step_size: int = 10, max_retries: int = 3, retry_delay: float = 1.0, 
                 max_workers: int = 8, graph_sequence_feature: Optional[str] = None):
        """
        Initialize RankGPT ranker.
        
        Args:
            api_base: API base URL (defaults to OPENAI_BASE_URL env var)
            api_key: API key (defaults to OPENAI_API_KEY env var)
            model_name: Model name to use for ranking
            window_size: Size of sliding window for ranking
            step_size: Step size for sliding window
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
            max_workers: Maximum number of worker threads
            graph_sequence_feature: Optional feature name for graph sequence 
                ('highlighted_determ_sequence' or 'no_highlighted_determ_sequence')
        """
        self.api_base = api_base or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "None")
        self.model_name = model_name
        self.window_size = window_size
        self.step_size = step_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_workers = max_workers
        self.graph_sequence_feature = graph_sequence_feature

        if not self.api_base:
            raise ValueError("API base URL must be provided via api_base parameter or OPENAI_BASE_URL environment variable")
        if not self.api_key:
            raise ValueError("API key must be provided via api_key parameter or OPENAI_API_KEY environment variable")
        
        if graph_sequence_feature and graph_sequence_feature not in ["highlighted_determ_sequence", "no_highlighted_determ_sequence"]:
            raise ValueError(f"graph_sequence_feature must be one of: 'highlighted_determ_sequence', 'no_highlighted_determ_sequence', or None")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    def fit(self, train_df: pd.DataFrame) -> None:
        """No-op for zero-shot ranking (no training required)"""
        pass

    def _process_group(self, question_id, group) -> RankedAnswersDict:
        # Extract unique answers for this question
        unique_answers = extract_unique_answers_from_group(group, self.graph_sequence_feature)
        
        if not unique_answers:
            # No answers to rank, use original order
            answers = list(dict.fromkeys(group["model_answers"].iloc[0]).keys())
            ranked_answers = self._model_answers_to_ranked_answers(answers)
        elif len(unique_answers) <= self.window_size:
            # Single-pass ranking
            ranked_answers = self._rank_single_pass(group["question"].iloc[0], unique_answers)
        else:
            # Sliding window ranking
            ranked_answers = self._sliding_window_rank(group["question"].iloc[0], unique_answers)
            
        return RankedAnswersDict(
            QuestionID=question_id,
            RankedAnswers=ranked_answers
        )
    
    def rerank(self, test_df: pd.DataFrame) -> List[RankedAnswersDict]:
        """
        Rerank answers using RankGPT approach.
        
        Args:
            test_df: DataFrame with test data
            
        Returns:
            List of ranked answers in the required format
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_group, question_id, group) for question_id, group in test_df.groupby("id")]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Ranking questions"):
                results.append(future.result())
        
        return results
    
    def _rank_single_pass(self, question: str, answers: List[Dict[str, Any]]) -> List[RankedAnswer]:
        """
        Rank all answers in a single API call.
        
        Args:
            question: The question to rank answers for
            answers: List of unique answer dictionaries
            
        Returns:
            List of ranked answers
        """
        prompt = create_ranking_prompt(question, answers)
        
        try:
            response = self._call_openai_api(prompt)
            ranking = parse_ranking_output(response, len(answers))

            if ranking is None:
                # Fallback to original order
                return self._answers_to_ranked_answers(answers)
            
            # Apply ranking
            ranked_answers = []
            for rank_idx in ranking:
                if 1 <= rank_idx <= len(answers):
                    answer = answers[rank_idx - 1]  # Convert to 0-based index
                    ranked_answers.append(
                        RankedAnswer(
                            AnswerEntityID=answer.get("entity_id"),
                            AnswerString=answer["answer_string"],
                            Score=None
                        )
                    )
            
            return ranked_answers
            
        except Exception as e:
            print(f"Error in single-pass ranking: {e}")
            return self._answers_to_ranked_answers(answers)
    
    def _sliding_window_rank(self, question: str, all_answers: List[Dict[str, Any]]) -> List[RankedAnswer]:
        """
        Rank answers using sliding window approach.
        
        Args:
            question: The question to rank answers for
            all_answers: List of all unique answer dictionaries
            
        Returns:
            List of ranked answers
        """
        num_answers = len(all_answers)
        scores = np.zeros(num_answers)
        
        # Create sliding windows
        for start_idx in range(0, num_answers, self.step_size):
            end_idx = min(start_idx + self.window_size, num_answers)
            window_answers = all_answers[start_idx:end_idx]
            
            if len(window_answers) < 2:
                continue
            
            prompt = create_ranking_prompt(question, window_answers, start_idx + 1)
            
            try:
                response = self._call_openai_api(prompt)
                ranking = parse_ranking_output(response, len(window_answers))

                
                if ranking is not None:
                    # Apply position-based scoring (position i gets score 1/i)
                    for rank_pos, rank_idx in enumerate(ranking):
                        if 1 <= rank_idx <= len(window_answers):
                            global_idx = start_idx + rank_idx - 1
                            if 0 <= global_idx < num_answers:
                                scores[global_idx] += 1.0 / (rank_pos + 1)
            
            except Exception as e:
                print(f"Error in sliding window ranking for window {start_idx}-{end_idx}: {e}")
                continue
        
        # Sort by scores (higher is better)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Create ranked answers
        ranked_answers = []
        for idx in sorted_indices:
            answer = all_answers[idx]
            ranked_answers.append(
                RankedAnswer(
                    AnswerEntityID=answer.get("entity_id"),
                    AnswerString=answer["answer_string"],
                    Score=float(scores[idx]) if scores[idx] > 0 else None
                )
            )
        
        return ranked_answers
    
    def _call_openai_api(self, prompt: str) -> str:
        """
        Make API call to OpenAI-compatible endpoint.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Response text from the API
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,  # Deterministic output
                    max_tokens=1000
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise e
        
        raise Exception(f"API call failed after {self.max_retries} attempts")
    
    def _answers_to_ranked_answers(self, answers: List[Dict[str, Any]]) -> List[RankedAnswer]:
        """Convert answer dictionaries to RankedAnswer format"""
        return [
            RankedAnswer(
                AnswerEntityID=answer.get("entity_id"),
                AnswerString=answer["answer_string"],
                Score=None
            )
            for answer in answers
        ]
    
    def _model_answers_to_ranked_answers(self, model_answers: List[str]) -> List[RankedAnswer]:
        """Convert model answers to RankedAnswer format (fallback)"""
        return [
            RankedAnswer(
                AnswerEntityID=None,
                AnswerString=answer,
                Score=None
            )
            for answer in model_answers
        ]
