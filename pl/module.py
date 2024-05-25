from typing import Any

import peft
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
import torchmetrics
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel

from config import Config


class TweetLM(pl.LightningModule):
    def __init__(self, pad_token_id=None):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(Config.name_or_path, trust_remote_code=True)
        self.pad_token_id = pad_token_id

        hidden_size = self.base_model.config.hidden_size
        self.sentiment_head = nn.Linear(in_features=hidden_size, out_features=Config.num_cls_sent, bias=True)
        self.stance_head = nn.Linear(in_features=hidden_size, out_features=Config.num_cls_stan, bias=True)

        l_config = LoraConfig(
            r=4,
            lora_alpha=4,
            lora_dropout=0.1,
            use_dora=True,
            use_rslora=True,
            target_modules=["q_proj", "v_proj"],
            task_type=peft.TaskType.FEATURE_EXTRACTION,
            bias="lora_only",
        )

        self.base_model = get_peft_model(self.base_model, l_config)
        print(self.base_model)

        self._init_metrics()

    def _init_metrics(self):
        self.stance_acc = torchmetrics.Accuracy(num_classes=Config.num_cls_stan, task="binary", average="macro")
        self.stance_precision = torchmetrics.Precision(num_classes=Config.num_cls_stan, average='macro',
                                                       zero_division=0, task="binary")
        self.stance_recall = torchmetrics.Recall(num_classes=Config.num_cls_stan, average='macro', zero_division=0,
                                                 task="binary")
        self.stance_f1 = torchmetrics.F1Score(num_classes=Config.num_cls_stan, average='macro', zero_division=0,
                                              task="binary")
        self.stance_kappa = torchmetrics.CohenKappa(num_classes=Config.num_cls_stan, task="binary")

        self.sentiment_acc = torchmetrics.Accuracy(num_classes=Config.num_cls_sent, task="multiclass", average="macro")
        self.sentiment_precision = torchmetrics.Precision(num_classes=Config.num_cls_sent, average='macro',
                                                          zero_division=0,
                                                          task="multiclass")
        self.sentiment_recall = torchmetrics.Recall(num_classes=Config.num_cls_sent, average='macro', zero_division=0,
                                                    task="multiclass")
        self.sentiment_f1 = torchmetrics.F1Score(num_classes=Config.num_cls_sent, average='macro', zero_division=0,
                                                 task="multiclass")
        self.sentiment_kappa = torchmetrics.CohenKappa(num_classes=Config.num_cls_sent, task="multiclass")

    def compute_loss(self, stance_logits, sentiment_logits, labels, sentiment_mask, stance_mask):
        weights = self.val_dataloaders[0].dataset.senti_data.class_weights
        criterion_stance = torch.nn.CrossEntropyLoss()
        criterion_sentiment = torch.nn.CrossEntropyLoss(weight=weights)

        stance_labels = labels[stance_mask.bool()]
        sentiment_labels = labels[sentiment_mask.bool()]

        stance_loss = 0
        if len(stance_labels) > 0:
            stance_loss = criterion_stance(stance_logits, stance_labels)

        sentiment_loss = 0
        if len(sentiment_labels) > 0:
            sentiment_loss = criterion_sentiment(sentiment_logits, sentiment_labels)

        total_loss = stance_loss + sentiment_loss
        stance_loss = stance_loss.item() if isinstance(stance_loss, torch.Tensor) else stance_loss
        sentiment_loss = sentiment_loss.item() if isinstance(sentiment_loss, torch.Tensor) else sentiment_loss
        return total_loss, stance_loss, sentiment_loss

    def forward(self, input_ids, attn_mask, task_mask=None):
        out = self.base_model(input_ids, attn_mask)
        hidden_states = out[0]
        stance_logits = self.stance_head(hidden_states)
        sentiment_logits = self.sentiment_head(hidden_states)

        batch_size = input_ids.shape[0]

        if input_ids is not None:
            sequence_lengths = torch.eq(input_ids, self.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths
        else:
            sequence_lengths = -1

        pooled_stan_logits = stance_logits[torch.arange(batch_size), sequence_lengths]
        pooled_sent_logits = sentiment_logits[torch.arange(batch_size), sequence_lengths]

        if task_mask is None:
            return pooled_stan_logits, pooled_sent_logits

        stance_mask, sentiment_mask = task_mask

        masked_stan_logits = pooled_stan_logits[stance_mask]
        masked_sent_logits = pooled_sent_logits[sentiment_mask]

        return masked_stan_logits, masked_sent_logits

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=Config.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
        train_batch = train_batch[0]
        input_ids = train_batch["input_ids"]
        attn_mask = train_batch["attention_mask"]
        sentiment_mask = train_batch["sentiment_mask"]
        stance_mask = train_batch["stance_mask"]
        labels = train_batch["labels"]

        stance_logits, sentiment_logits = self.forward(input_ids, attn_mask, (stance_mask, sentiment_mask))

        loss, stance_loss, sent_loss = self.compute_loss(stance_logits, sentiment_logits, labels, sentiment_mask,
                                                         stance_mask)

        losses = {
            "train_total_loss": loss,
            "train_stance_loss": stance_loss,
            "train_sentiment_loss": sent_loss
        }
        self.log_dict(losses, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx) -> STEP_OUTPUT:
        input_ids = val_batch["input_ids"]
        attn_mask = val_batch["attention_mask"]
        sentiment_mask = val_batch["sentiment_mask"]
        stance_mask = val_batch["stance_mask"]
        labels = val_batch["labels"]

        stance_logits, sentiment_logits = self.forward(input_ids, attn_mask, (stance_mask, sentiment_mask))

        loss, stance_loss, sent_loss = self.compute_loss(stance_logits, sentiment_logits, labels, sentiment_mask,
                                                         stance_mask)
        stance_preds = torch.argmax(stance_logits, dim=1)
        sentiment_preds = torch.argmax(sentiment_logits, dim=1)
        stance_labels = labels[stance_mask]
        sentiment_labels = labels[sentiment_mask]

        stance_metrics = {}
        sentiment_metrics = {}
        if len(stance_labels) > 0:
            stance_metrics = {
                "val/stance_accuracy": self.stance_acc(stance_preds, stance_labels),
                "val/stance_precision": self.stance_precision(stance_preds, stance_labels),
                "val/stance_recall": self.stance_recall(stance_preds, stance_labels),
                "val/stance_f1": self.stance_f1(stance_preds, stance_labels),
                "val/stance_kappa": self.stance_kappa(stance_preds, stance_labels),
            }
        if len(sentiment_labels) > 0:
            sentiment_metrics = {
                "val/sentiment_accuracy": self.sentiment_acc(sentiment_preds, sentiment_labels),
                "val/sentiment_precision": self.sentiment_precision(sentiment_preds, sentiment_labels),
                "val/sentiment_recall": self.sentiment_recall(sentiment_preds, sentiment_labels),
                "val/sentiment_f1": self.sentiment_f1(sentiment_preds, sentiment_labels),
                "val/sentiment_kappa": self.sentiment_kappa(sentiment_preds, sentiment_labels),

            }

        losses = {
            "val_total_loss": loss,
            "val_stance_loss": stance_loss,
            "val_sentiment_loss": sent_loss,
        }
        self.log_dict(losses, on_step=True, on_epoch=True)

        metrics = {
            **stance_metrics,
            **sentiment_metrics
        }
        self.log_dict(metrics, on_step=False, on_epoch=True)
