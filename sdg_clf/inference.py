import numpy as np
import torch
from scipy import special

from sdg_clf import utils


class TextToModelInputs:
    def __init__(self, tokenizer):
        self.max_length = 260
        self.tokenizer = tokenizer

    def prepare_input_ids(self, text: str) -> torch.Tensor:
        input_ids = self.tokenizer(text, add_special_tokens=False)
        input_ids = input_ids["input_ids"]
        input_ids = [input_ids[x:x + self.max_length - 2] for x in range(0, len(input_ids), self.max_length)]
        # add bos and eos tokens to each input_ids
        input_ids = [[self.tokenizer.cls_token_id] + x + [self.tokenizer.sep_token_id] for x in input_ids]
        # pad input_ids to max_length
        input_ids[-1] = input_ids[-1] + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids[-1]))
        return torch.tensor(input_ids)

    def prepare_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows, columns = input_ids.shape
        attention_mask = torch.ones((rows - 1, columns))
        last_mask = torch.tensor([[1 if id != self.tokenizer.pad_token_id else 0 for id in input_ids[-1, :]]])
        attention_mask = torch.concat((attention_mask, last_mask), dim=0)
        return attention_mask

    def prepare_model_inputs(self, text: str) -> dict[str, torch.Tensor]:
        input_ids = self.prepare_input_ids(text)
        attention_mask = self.prepare_attention_mask(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def predict_on_sample(text: str, model, tokenizer) -> list[int]:
    text = utils.process_text(text)
    model_inputs = TextToModelInputs(tokenizer).prepare_model_inputs(text)
    model_inputs = {k: v.numpy() for k, v in model_inputs.items()}
    logits = model.run(None, model_inputs)[0]
    # apply sigmoid
    outputs = special.expit(logits)
    # apply threshold
    outputs = outputs > 0.27
    outputs = np.any(outputs, axis=0)
    # get indices
    indices = outputs.nonzero()[0]
    sdgs = indices + 1
    sdgs = sdgs.tolist()
    return sdgs

