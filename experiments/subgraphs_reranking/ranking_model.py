"""model class for all rerankers approach"""
import random
from abc import ABC, abstractmethod
from typing import List, TypedDict, Optional
import joblib
import numpy as np
import torch
from pandas import DataFrame
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from catboost import CatBoostRegressor
from ranking_data_utils import df_to_features_array


class RankedAnswer(TypedDict):
    """each answer format"""

    AnswerEntityID: str
    AnswerString: str
    Score: float


class RankedAnswersDict(TypedDict):
    """each question format (multiple ranked answer)"""

    QuestionID: str
    RankedAnswers: List[RankedAnswer]


class Ranker(ABC):
    """abstract ranker class, all approaches follow this format"""

    @abstractmethod
    def fit(self, train_df: DataFrame) -> None:
        """fit the model if needed"""
        return

    @abstractmethod
    def rerank(self, test_df: DataFrame) -> List[RankedAnswersDict]:
        """rerank the fit/trained model"""
        return


class NORanker(Ranker):
    """no reranking, take the base LLM's output"""

    def fit(self, train_df: DataFrame) -> None:
        pass

    def rerank(self, test_df: DataFrame) -> List[RankedAnswersDict]:
        test_df = test_df.drop_duplicates(subset=["id"])
        results = []
        for _, row in test_df.iterrows():
            answers = list(dict.fromkeys(row["model_answers"]).keys())

            ranked_answers = [
                RankedAnswer(
                    AnswerEntityID=None,
                    AnswerString=answer,
                    Score=None,
                )
                for answer in answers
            ]
            results.append(
                RankedAnswersDict(
                    QuestionID=row["id"],
                    RankedAnswers=ranked_answers,
                )
            )
        return results


class FullRandomRanker(Ranker):
    """random ranking"""

    def fit(self, train_df: DataFrame) -> None:
        pass

    def rerank(self, test_df: DataFrame) -> List[RankedAnswersDict]:
        """rerank the answers by shuffling the base LLM outputs"""
        test_df = test_df.drop_duplicates(subset=["id"])
        results = []
        for _, row in test_df.iterrows():
            answers = list(set(row["model_answers"]))
            random.shuffle(answers)

            ranked_answers = [
                RankedAnswer(
                    AnswerEntityID=None,
                    AnswerString=answer,
                    Score=None,
                )
                for answer in answers
            ]
            results.append(
                RankedAnswersDict(
                    QuestionID=row["id"],
                    RankedAnswers=ranked_answers,
                )
            )
        return results


class RankerBase(Ranker):
    """base class for all reranker to inherit"""

    def fit(self, train_df: DataFrame) -> None:
        raise NotImplementedError()

    def rerank(self, test_df: DataFrame) -> List[RankedAnswersDict]:
        raise NotImplementedError()

    def _sort_answers_group_by_scores(
        self, group: DataFrame, scores: np.ndarray
    ) -> List[RankedAnswer]:
        sorted_scores_idxs = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_scores_idxs]
        sorted_ranked_answers = group["answerEntity"].values[sorted_scores_idxs]

        ranked_answers = []
        for score, answer_entity_id in zip(sorted_scores, sorted_ranked_answers):
            ranked_answers.append(
                RankedAnswer(
                    AnswerEntityID=answer_entity_id, AnswerString=None, Score=score
                )
            )
        return ranked_answers

    def _model_answers_to_ranked_answers(
        self, model_answers: List[str]
    ) -> List[RankedAnswer]:
        """build the dict of ranked answer"""
        return [
            RankedAnswer(
                AnswerEntityID=None,
                AnswerString=answer,
                Score=None,
            )
            for answer in model_answers
        ]


class LogisticRegressionRanker(RankerBase):
    """reranker for LogisticRegression"""

    def __init__(
        self,
        sequence_features: Optional[list] = None,
        graph_features: Optional[list] = None,
    ):
        """sequence features q_a and/or g2t_determ, g2t_t5, g2t"""
        self.sequence_features = sequence_features if sequence_features else []
        self.graph_features = graph_features if graph_features else []
        self.features_to_use = self.sequence_features + self.graph_features
        self.model = None
        self.fitted_scaler = None

    def fit(self, train_df: DataFrame, **kwargs) -> None:
        """fit LogReg on train_df"""
        train_df = train_df.dropna(subset=["graph"])

        #  scale graph features if we scaler arg is not none
        scaler = kwargs.get("scaler")
        if scaler:
            if len(self.graph_features) == 0:
                raise ValueError(
                    "Scaler indicated with no numeric/graph features to scale."
                )
            train_df[self.graph_features] = scaler.fit_transform(
                train_df[self.graph_features]
            )
            self.fitted_scaler = scaler  # save fitted scaler

        x_train = df_to_features_array(train_df[self.features_to_use])
        y_train = train_df["correct"].astype(np.float16)

        self.model = LogisticRegression(
            class_weight="balanced",
            n_jobs=kwargs.get("n_jobs"),
            max_iter=kwargs.get("max_iter", 1000),
        )
        self.model.fit(x_train, y_train)

    def rerank(self, test_df: DataFrame) -> List[RankedAnswersDict]:
        """given test_df, rerank using the fitted model from func above and
        output the predictions in the specified format above"""
        if self.model is None:
            raise NotFittedError("This ranker model is not fitted yet.")

        if self.fitted_scaler:  # fit if we have a scaler
            test_df[self.graph_features] = self.fitted_scaler.transform(
                test_df[self.graph_features]
            )

        results = []
        for question_id, group in test_df.groupby("id"):
            if isinstance(group["graph"].iloc[0], (dict, str)):
                # If we have subgraphs, use it to rerank
                group_features = df_to_features_array(group[self.features_to_use])
                scores = self.model.predict_proba(group_features)[:, 1]
                ranked_answers = self._sort_answers_group_by_scores(group, scores)
            else:
                # If we have no subgraphs, just use initial answers
                answers = list(dict.fromkeys(group["model_answers"].iloc[0]).keys())
                ranked_answers = self._model_answers_to_ranked_answers(answers)

            results.append(
                RankedAnswersDict(
                    QuestionID=question_id,
                    RankedAnswers=ranked_answers,
                )
            )
        return results


class LinearRegressionRanker(RankerBase):
    """reranker for LinearRegression"""

    def __init__(
        self,
        sequence_features: Optional[list] = None,
        graph_features: Optional[list] = None,
    ):
        self.sequence_features = sequence_features if sequence_features else []
        self.graph_features = graph_features if graph_features else []
        self.features_to_use = self.sequence_features + self.graph_features
        self.model = None
        self.fitted_scaler = None

    def fit(self, train_df: DataFrame, **kwargs) -> None:
        """fit LinReg on train_df"""
        train_df = train_df.dropna(subset=["graph"])

        #  scale graph features if we scaler arg is not none
        scaler = kwargs.get("scaler")
        if scaler:
            if len(self.graph_features) == 0:
                raise ValueError(
                    "Scaler indicated with no numeric/graph features to scale."
                )
            train_df[self.graph_features] = scaler.fit_transform(
                train_df[self.graph_features]
            )
            self.fitted_scaler = scaler  # save fitted scaler

        x_train = df_to_features_array(train_df[self.features_to_use])
        y_train = train_df["correct"].astype(np.float16)

        self.model = LinearRegression(
            n_jobs=kwargs.get("n_jobs"),
        )
        self.model.fit(x_train, y_train)

    def rerank(self, test_df: DataFrame) -> List[RankedAnswersDict]:
        """given test_df, rerank using the fitted model from func above and
        output the predictions in the specified format above"""
        if self.model is None:
            raise NotFittedError("This ranker model is not fitted yet.")

        if self.fitted_scaler:  # fit graph features if we have a scaler
            test_df[self.graph_features] = self.fitted_scaler.transform(
                test_df[self.graph_features]
            )

        results = []
        for question_id, group in test_df.groupby("id"):
            if isinstance(group["graph"].iloc[0], (dict, str)):
                # If we have subgraphs, use it to rerank
                group_features = df_to_features_array(group[self.features_to_use])
                scores = self.model.predict(group_features)
                ranked_answers = self._sort_answers_group_by_scores(group, scores)
            else:
                # If we have no subgraphs, just use initial answers
                answers = list(dict.fromkeys(group["model_answers"].iloc[0]).keys())
                ranked_answers = self._model_answers_to_ranked_answers(answers)

            results.append(
                RankedAnswersDict(
                    QuestionID=question_id,
                    RankedAnswers=ranked_answers,
                )
            )
        return results


class MPNetRanker(RankerBase):
    """reranker for MPNET"""

    # pylint: disable=no-member
    def __init__(self, feature_to_use: str, model_path: str, device: torch.device):
        self.feature_to_use = feature_to_use
        self.model_path = model_path

        self.device = device
        self.tokenizer = None
        self.model = None

        try:
            print("Trying to load the model...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            ).to(device)
            print("Model Loaded.")
        except Exception as exception:  # pylint: disable=broad-except
            print(f"Failed to load model: {exception}")

    def fit(self, train_df: DataFrame, **kwargs) -> None:
        raise NotImplementedError(
            "No fit function for MPNET. Model should be trained already."
        )

    def rerank(self, test_df: DataFrame) -> List[RankedAnswersDict]:
        """given test_df, rerank using the trained model and output the
        predictions in the specified format above"""
        if self.model is None or self.tokenizer is None:
            raise NotFittedError("This ranker model is not fitted yet.")

        results = []
        groups = test_df.groupby("id")
        for question_id, group in tqdm(groups, total=len(test_df["id"].unique())):
            if isinstance(group["graph"].iloc[0], (dict, str)):
                # If we have subgraphs, use it to rerank
                with torch.no_grad():
                    tokenized_data = self.tokenizer(
                        text=group[self.feature_to_use].values.tolist(),
                        padding="max_length",
                        max_length=512,
                        truncation=True,
                        return_tensors="pt",
                    )
                    outputs = self.model(
                        input_ids=tokenized_data["input_ids"].to(self.device),
                        attention_mask=tokenized_data["attention_mask"].to(self.device),
                    )
                    scores = (
                        outputs.logits.cpu().detach().flatten().numpy().astype(float)
                    )

                ranked_answers = self._sort_answers_group_by_scores(group, scores)
            else:
                # If we have no subgraphs, just use initial answers
                answers = list(dict.fromkeys(group["model_answers"].iloc[0]).keys())
                ranked_answers = self._model_answers_to_ranked_answers(answers)

            results.append(
                RankedAnswersDict(
                    QuestionID=question_id,
                    RankedAnswers=ranked_answers,
                )
            )
        return results


class CatboostRanker(RankerBase):
    """reranker for Catboost"""

    def __init__(
        self,
        model_path,
        sequence_features: Optional[list] = None,
        graph_features: Optional[list] = None,
        scaler_path: Optional[str] = None,
    ):
        self.sequence_features = sequence_features if sequence_features else []
        self.graph_features = graph_features if graph_features else []
        self.features_to_use = self.sequence_features + self.graph_features

        self.model = None
        try:
            print("Trying to load the model...")
            self.model = CatBoostRegressor().load_model(model_path)
            print("Model Loaded.")
        except Exception as exception:  # pylint: disable=broad-except
            print(f"Failed to load model: {exception}")

        self.fitted_scaler = None
        if scaler_path:
            try:
                print("Trying to load the fitted scaler...")
                self.fitted_scaler = joblib.load(scaler_path)
                if len(self.graph_features) == 0:
                    raise ValueError(
                        "Scaler indicated with no numeric/graph features to scale."
                    )
            except Exception as exception:  # pylint: disable=broad-except
                print(f"Failed to load fitted scaler: {exception}")

    def fit(self, train_df: DataFrame, **kwargs) -> None:
        raise NotImplementedError(
            "No fit function for CatBoost. Model should be trained already."
        )

    def rerank(self, test_df: DataFrame) -> List[RankedAnswersDict]:
        """given test_df, rerank using the trained model and output the
        predictions in the specified format above"""
        if self.model is None:
            raise NotFittedError("This ranker model is not fitted yet.")

        if self.fitted_scaler:  # fit graph features if we have a scaler
            test_df[self.graph_features] = self.fitted_scaler.transform(
                test_df[self.graph_features]
            )

        results = []
        groups = test_df.groupby("id")
        for question_id, group in tqdm(groups, total=len(test_df["id"].unique())):
            if isinstance(group["graph"].iloc[0], (dict, str)):
                # If we have subgraphs, use it to rerank
                scores = self.model.predict(group[self.features_to_use])
                ranked_answers = self._sort_answers_group_by_scores(group, scores)
            else:
                # If we have no subgraphs, just use initial answers
                answers = list(dict.fromkeys(group["model_answers"].iloc[0]).keys())
                ranked_answers = self._model_answers_to_ranked_answers(answers)

            results.append(
                RankedAnswersDict(
                    QuestionID=question_id,
                    RankedAnswers=ranked_answers,
                )
            )
        return results
