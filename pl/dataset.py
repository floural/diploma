from enum import Enum
from typing import Any, List, Dict, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd

from config import Config


class TaskType(Enum):
    SENTIMENT = 0
    STANCE = 1


class SentimentDataset(Dataset):
    def __init__(self, file_path, tokenizer) -> None:
        super().__init__()
        self.data = pd.read_json(file_path, lines=True)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Any:
        item = self.data.iloc[index]
        out = {
            "question": "",
            "text": item["text"],
            "label": item["gold_label"],
            "topic": item["target"]
        }
        return out


class StanceDataset(Dataset):
    def __init__(self, data_path, tokenizer) -> None:
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Any:
        item = self.data.iloc[index]
        out = {
            "question": item["question"],
            "text": item["comment"],
            "topic": item["topic"],
            "label": item["label"]
        }
        return out


class MergedDataset(Dataset):
    def __init__(self, senti_data: SentimentDataset, stan_data: StanceDataset, is_train: bool) -> None:
        super().__init__()
        self.senti_data = senti_data
        self.stan_data = stan_data
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.senti_data) + len(self.stan_data)
    
    def __getitem__(self, index) -> Any:
        item = None
        if self.is_train:
            if index % 2 == 0:
                item = self.senti_data[(index // 2) % len(self.senti_data)]
                task_type = TaskType.SENTIMENT
            else:
                item = self.stan_data[(index // 2) % len(self.stan_data)]
                task_type = TaskType.STANCE
        else:
            if index > len(self.senti_data) - 1:
                item = self.stan_data[index - len(self.senti_data)]
                task_type = TaskType.STANCE
            else:
                item = self.senti_data[index]
                task_type = TaskType.SENTIMENT
        
        out = {
            "question": item["question"],
            "text": item["text"],
            "topic": item["topic"],
            "label": item["label"],
            "task_type": task_type
        }
        return out


class DefaultCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        questions = [feature['question'] for feature in features]
        texts = [feature['text'] for feature in features]
        topics = [feature['topic'] for feature in features]
        labels = [feature['label'] for feature in features]
        sentiment_mask = [1 if feature["task_type"] == TaskType.SENTIMENT else 0 for feature in features]
        stance_mask = [1 if feature["task_type"] == TaskType.STANCE else 0 for feature in features]

        inputs = self.tokenizer(topics, texts, questions, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        inputs["sentiment_mask"] = torch.tensor(sentiment_mask, dtype=torch.bool)
        inputs["stance_mask"] = torch.tensor(stance_mask, dtype=torch.bool)

        return inputs


class DataLoaderFactory:
    _tokenizer = AutoTokenizer.from_pretrained(Config.name_or_path, trust_remote_code=True)

    @staticmethod
    def _create_loader(dataset: Dataset, batch_size: int, is_train: bool) -> DataLoader:
        """Creates a DataLoader for a given dataset."""
        collator = DefaultCollator(DataLoaderFactory._tokenizer)
        shuffle = True if is_train else False
        persistent_workers = True if is_train else False
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

    @staticmethod
    def get_sentiment_dataloader(sentiment_data_path: str, is_train: bool) -> DataLoader:
        """Creates a DataLoader for the sentiment dataset using configuration settings."""
        sentiment_dataset = SentimentDataset(sentiment_data_path)
        return DataLoaderFactory._create_loader(sentiment_dataset, Config.batch_size, is_train=is_train)

    @staticmethod
    def get_stance_dataloader(stance_data_path: str, is_train: bool) -> DataLoader:
        """Creates a DataLoader for the stance dataset using configuration settings."""
        stance_dataset = StanceDataset(stance_data_path)
        return DataLoaderFactory._create_loader(stance_dataset, Config.batch_size, is_train=is_train)

    @staticmethod
    def get_merged_dataloader(sentiment_data_path: str, stance_data_path: str, is_train: bool) -> DataLoader:
        """Creates a DataLoader for the merged dataset of sentiment and stance using configuration settings."""
        sentiment_dataset = SentimentDataset(sentiment_data_path)
        stance_dataset = StanceDataset(stance_data_path)
        merged_dataset = MergedDataset(sentiment_dataset, stance_dataset, is_train)
        return DataLoaderFactory._create_loader(merged_dataset, Config.batch_size, is_train=is_train)
