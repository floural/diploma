import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config


class TweetLM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(Config.name_or_path, trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(Config.name_or_path, trust_remote_code=True).model
        self.base_model.config.pad_token_id = tokenizer.eos_token_id

        hidden_size = self.base_model.config.hidden_size
        self.sentiment_head = nn.Linear(in_features=hidden_size, out_features=Config.num_cls_sent, bias=True)
        self.stance_head = nn.Linear(in_features=hidden_size, out_features=Config.num_cls_stan, bias=True)

    def forward(self, input_ids, attn_mask, task_mask=None):
        out = self.base_model(input_ids, attn_mask)
        hidden_states = out[0]
        stance_logits = self.stance_head(hidden_states)
        sentiment_logits = self.sentiment_head(hidden_states)

        batch_size = input_ids.shape[0]

        if input_ids is not None:
            sequence_lengths = torch.eq(input_ids, self.base_model.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(stance_logits.device)
        else:
            sequence_lengths = -1

        pooled_stan_logits = stance_logits[torch.arange(batch_size, device=stance_logits.device), sequence_lengths]
        pooled_sent_logits = sentiment_logits[torch.arange(batch_size, device=stance_logits.device), sequence_lengths]

        if task_mask is None:
            return pooled_stan_logits, pooled_sent_logits

        stance_mask, sentiment_mask = task_mask

        masked_stan_logits = pooled_stan_logits[stance_mask]
        masked_sent_logits = pooled_sent_logits[sentiment_mask]

        return masked_stan_logits, masked_sent_logits
