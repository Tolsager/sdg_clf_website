import numpy as np
from scipy import special

from sdg_clf import utils


class TextToModelInputs:
    def __init__(self, tokenizer):
        self.max_length = 260
        self.tokenizer = tokenizer

    def prepare_input_ids(self, text: str) -> list[np.ndarray]:
        input_ids = self.tokenizer(text, add_special_tokens=False)
        input_ids = input_ids["input_ids"]
        input_ids = [input_ids[x:x + self.max_length - 2] for x in range(0, len(input_ids), self.max_length)]
        # add bos and eos tokens to each input_ids
        input_ids = [[self.tokenizer.cls_token_id] + x + [self.tokenizer.sep_token_id] for x in input_ids]
        # pad input_ids to max_length
        input_ids[-1] = input_ids[-1] + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids[-1]))
        # convert sublists to arrays
        input_ids = [np.array([x]) for x in input_ids]

        return input_ids

    def prepare_attention_mask(self, input_ids: list[np.ndarray]) -> list[np.ndarray]:
        n_samples = len(input_ids)
        attention_mask = [np.ones((1, self.max_length)) for _ in range(n_samples-1)]
        last_mask = np.array([[1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids[-1][0, :]]])
        attention_mask.append(last_mask)
        return attention_mask

    def prepare_model_inputs(self, text: str) -> dict[str, list[np.ndarray]]:
        input_ids = self.prepare_input_ids(text)
        attention_mask = self.prepare_attention_mask(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def predict_on_sample(text: str, model, tokenizer) -> list[int]:
    text = utils.process_text(text)
    model_inputs = TextToModelInputs(tokenizer).prepare_model_inputs(text)
    # get logits from each sample in model_inputs
    all_logits = []
    for ids, mask in zip(model_inputs["input_ids"], model_inputs["attention_mask"]):
        # TODO: remake onnx model with int as input types and enable variable batch sizes
        current_model_inputs = {"input_ids": ids.astype(int), "attention_mask": mask.astype(np.float32)}
        logits = model.run(None, current_model_inputs)[0]
        all_logits.append(logits)
    all_logits = np.concatenate(all_logits, axis=0)
    # apply sigmoid
    outputs = special.expit(all_logits)
    # apply threshold
    outputs = outputs > 0.27
    outputs = np.any(outputs, axis=0)
    # get indices
    indices = outputs.nonzero()[0]
    sdgs = indices + 1
    sdgs = sdgs.tolist()
    return sdgs
