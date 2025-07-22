
import torch

class MyProcessingClass:
    def __init__(self, solver):
        self.solver = solver
        self.base_tokenizer = solver.tokenizer
        self.device = solver.device
    
    def __call__(self, text, **kwargs):
        # the parameter name 'text' just means preprocessed dataset
        # if 'text' is type of list, it's batched input
        # each datapoint is dictionary

        data = text
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list): assert False

        # we ignore other kwargs in GRPOTrainer
        # return batched input_ids
        return {
            "input_ids": torch.tensor([ d["input_ids"] for d in data ]).to(self.device),
            "attention_mask": torch.tensor([ d["attention_mask"] for d in data ]).to(self.device)
        }

    def batch_decode(self, input_ids, **kwargs):
        return self.base_tokenizer.batch_decode(input_ids, **kwargs)

    @property
    def pad_token(self):
        return self.base_tokenizer.pad_token

    @property
    def eos_token(self):
        return self.base_tokenizer.eos_token

    @property
    def bos_token(self):
        return self.base_tokenizer.bos_token

    @property
    def pad_token_id(self):
        return self.base_tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.base_tokenizer.eos_token_id

    @property
    def bos_token_id(self):
        return self.base_tokenizer.bos_token_id