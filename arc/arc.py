import torch
from transformers import GenerationConfig, TrainingArguments, Trainer
from trl import SFTTrainer, SFTConfig
from typing import List
import numpy as np

from .prompt import *
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from datasets import Dataset
import datetime

from accelerate import Accelerator, PartialState

class ARCSolver:
    """
    You should implement a `Solver` class for the project.
    """

    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct", hf_token=None):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        self.config_path = "artifacts/config/config.yml"
        self.model_id = model_id
        self.hf_token = hf_token
        # self.accelerator = Accelerator(kwargs_handlers = [InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))])
        #if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
        #    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
        )
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.system_header = self.tokenizer.encode("<|im_start|>system\n", add_special_tokens=False)
        self.user_header = self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
        self.assist_header = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.end_tail = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        
        self.rng = np.random.default_rng(24)
        self.pixel_ids = [ self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10) ]
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def parse_grid(self, ids: List[int]):
        """
        Parse LLM generated sequence into ARC grid format
        Args: ids (List[int]): LLM generated token list
        Returns: grid (List[List[int]]): parsed 2D grid
        """
        grid = []
        row = []
        inv_map = {k: i for i, k in enumerate(self.pixel_ids)}

        for idx in ids:
            if idx == self.sep:
                if len(row) > 0:
                    grid.append(row.copy())
                    row.clear()
            else:
                if idx not in inv_map: continue
                row.append(inv_map.get(idx, 0))
        return grid
    
    def format_grid(self, grid):
        """
        Format 2D grid into LLM input tokens
        Args: grid (List[List[int]]): 2D grid
        Returns: ids (List[int]): Token list for LLM
        """
        ids = []
        for row in grid:
            for col in row:
                ids.append(self.pixel_ids[col])
            ids.append(self.sep)
        return ids
    

    def prompt_id(self, datapoint, train=False, reasoning=False):
        """
        datapoint (dict): 
            contains training data, test input
        prompt (dict): string prompt
        """
        training_data = datapoint['train']
        self.color_permu = self.rng.permutation(10)
        self.rot = self.rng.choice(4)
        self.flip = self.rng.choice(3)

        def transform(grid: List[List[int]]):
            grid = np.array(grid)
            grid = self.color_permu[grid]
            grid = np.rot90(grid, self.rot)
            if self.flip != 2: grid = np.flip(grid, axis=self.flip)
            return grid

        sys = self.system_header + self.tokenizer.encode(system_message+"\n", add_special_tokens=False) + self.end_tail
        user = self.user_header + self.tokenizer.encode(user_message_template1+"\n", add_special_tokens=False)
        input_head = self.tokenizer.encode("input:\n", add_special_tokens=False)
        output_head = self.tokenizer.encode("output:\n", add_special_tokens=False)
        for ex in training_data:
            input = transform(ex['input'])
            output = transform(ex['output'])       
            user += input_head + self.format_grid(input)
            user += output_head + self.format_grid(output) + [self.sep]

        test_data = datapoint['test'][0]
        input_test_data = transform(test_data['input'])
        
        user += self.tokenizer.encode("\n"+user_message_template2+"\n", add_special_tokens=False)
        user += input_head + self.format_grid(input_test_data)
        user += self.tokenizer.encode("\n"+user_message_template3+"\n", add_special_tokens=False) + self.end_tail
        
        assist = self.assist_header.copy()
        if train:
            output_test_data = transform(test_data['output'])
            assist += output_head + self.format_grid(output_test_data)
        prompt = sys + user + assist + output_head
        return {"input_ids": prompt}


    def preprocess_data(self, train_dataset):
        # generate and encode prompts 
        from statistics import quantiles, mean
        format_data = []
        token_lens = []
        for train_data in train_dataset:
            prompt = self.prompt_id(train_data, train=True)
            tokenized = self.tokenizer.pad(
                prompt, 
                return_tensors="pt",
                padding="max_length",
                max_length=1024
            )
            input_ids = tokenized.input_ids.squeeze()
            attn_mask = tokenized.attention_mask.squeeze()
            header_ids = self.assist_header #self.tokenizer(self.assist_header, add_special_tokens=False)["input_ids"]
            #print(self.tokenizer.decode(header_ids))

            input_ids_list = input_ids.tolist()
            num_tokens = attn_mask.sum().item()
            token_lens.append(num_tokens)
            if num_tokens >= len(input_ids_list): continue # truncate long sequences
            
            header_len = len(header_ids)
            found = False
            for i in range(len(input_ids_list) - header_len+1):
                if input_ids_list[i:i+header_len] == header_ids:
                    label_idx = i + header_len
                    found = True
            if not found: assert False # check if prompt is successfully created (maybe deprecated)
            
            labels = torch.full_like(input_ids, -100)
            labels[label_idx:] = input_ids[label_idx:]
            
            format_data.append({
                "input_ids": input_ids.tolist(),
                "attention_mask": attn_mask.tolist(),
                "labels": labels.tolist()
            })
        # Q = quantiles(token_lens, n=10)
        # print(min(token_lens), Q, max(token_lens))
        print(f"After truncating, there are {len(format_data)} rows left in the dataset.")
        dataset = Dataset.from_list(format_data)
        return dataset
        

    def train(self, train_dataset, adapter_name=None, checkpoint=None):
        """
        Train a model with train_dataset. 
        Currently using QLoRA = Quantization + LoRA
        Read a project documentation for a description of `examples` and `question`.
        """
        # example
        datapoint = train_dataset[0]
        prompt = self.prompt_id(datapoint, train=True)
        # print(prompt)

        # generate and encode prompts 
        dataset = self.preprocess_data(train_dataset)
        
        # create new LoRA finetuning adapter with adapter_name
        lora_config = LoraConfig(
            r=16, # 8 or 16
            lora_alpha=32, # 16 or 32
            target_modules=[
                "q_proj", "v_proj", "o_proj", "k_proj",
                "up_proj", "down_proj", "gate_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        if adapter_name is None: 
            adapter_name = str(datetime.datetime.now()).split('.')[0].replace(":","").replace("-","").replace(" ","_")
        save_directory = "./artifacts/" + adapter_name

        if checkpoint is None: 
            self.model = get_peft_model(self.model, lora_config)
        else: self.model = PeftModel.from_pretrained(self.model, "./artifacts/" + checkpoint, is_trainable=True)
        
        # self.accelerator.prepare(self.model)
        self.model.train()
        self.model.print_trainable_parameters()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        # Create Trainer and start training
        training_args = SFTConfig(
            output_dir=save_directory,
            per_device_train_batch_size=2, # set smaller when CUDA out of memory
            gradient_accumulation_steps=8, # set larger when CUDA out of memory
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False},
            num_train_epochs=1,
            learning_rate=1e-4,
            lr_scheduler_type="cosine", # try using lr scheduler
            warmup_ratio=0.1,
            #warmup_steps=50,
            weight_decay=0.0005,
            fp16=True, # try using bf16
            save_strategy="epoch",
            logging_steps=50,
            remove_unused_columns=False,
            max_seq_length=2048,
            ddp_find_unused_parameters=False,
            # completion_only_loss=True
        )
        
        def data_collator(features):
            return {
                "input_ids": torch.tensor([ f["input_ids"] for f in features ]),
                "attention_mask": torch.tensor([ f["attention_mask"] for f in features ]),
                "labels": torch.tensor([ f["labels"] for f in features ])
            }

        trainer = SFTTrainer(
            model=self.model, 
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        trainer.train()
        trainer.save_model(save_directory + "/checkpoint-final")


    def predict(self, datapoint):
        """
        A single example of test data is given.
        You should predict 2D grid (List[List[int]] or np.ndarray)
        """
        prompt = self.prompt_id(datapoint)
        input_ids = torch.tensor(prompt["input_ids"]).to(self.device) # tokenized.input_ids.squeeze().to(self.device)
        prompt_str = self.tokenizer.decode(input_ids.tolist())
        input_ids = input_ids.unsqueeze(0)
        # print(prompt_str)

        config = GenerationConfig(
            do_sample=False, # greedy sampling
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=512,
            # temperature=0.6, # other sampling techs
            # top_k=20,
            # top_p=0.95
        )

        # Generate the output
        output = self.model.generate(
            input_ids=input_ids,
            generation_config=config,
        ).squeeze().cpu()
        N_prompt = input_ids.numel()

        output = output[N_prompt:].tolist()
        # print(self.tokenizer.decode(output))

        # Parse the output
        try: 
            grid = np.array(self.parse_grid(output))
            # Inverse transformation
            inv_permu = np.empty_like(self.color_permu)
            inv_permu[self.color_permu] = np.arange(10)
            grid = inv_permu[grid]
            if self.flip != 2: 
                grid = np.flip(grid, axis=self.flip)
            # print(grid, self.rot)
            grid = np.rot90(grid, -self.rot)
        except: grid = np.zeros((3,3))
        
        # print(grid)
        return grid

    def predict_beam(self, datapoint):
        """
        predict with multiple candidates.
        """

        prompt = self.prompt_id(datapoint)
        input_ids = torch.tensor(prompt["input_ids"]).to(self.device) # tokenized.input_ids.squeeze().to(self.device)
        prompt_str = self.tokenizer.decode(input_ids.tolist())
        input_ids = input_ids.unsqueeze(0)
        # print(prompt_str)

        config = GenerationConfig(
            do_sample=False, # greedy sampling
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=512,
            num_beams=20,
            num_return_sequences=20,
            # return_dict_in_generate=True,
            # output_scores=True
        )

        # Generate the output
        N_prompt = input_ids.numel()
        output = self.model.generate(
            input_ids=input_ids,
            generation_config=config,
        )

        # log_probs = self.model.compute_transition_scores(
        #     output.sequences, output.scores, normalize_logits=True
        # ).sum(dim=1)
        # print(torch.exp(log_probs).cpu().tolist())

        grids = []
        for seq in output:
            seq = seq[N_prompt:].tolist()
            # print(self.tokenizer.decode(seq))
            try: # Parse the output
                grid = np.array(self.parse_grid(seq))
                # Inverse transformation
                inv_permu = np.empty_like(self.color_permu)
                inv_permu[self.color_permu] = np.arange(10)
                grid = inv_permu[grid]
                if self.flip != 2: 
                    grid = np.flip(grid, axis=self.flip)
                # print(grid, self.rot)
                grid = np.rot90(grid, -self.rot)
            except: grid = np.zeros((3,3))
            grids.append(grid)
            # print(grid)
        # print(torch.exp(output['sequences_scores']).cpu().tolist())
        return grids


    def predict_bfs(self, datapoint):

        prompt = self.prompt_id(datapoint)
        input_ids = torch.tensor(prompt["input_ids"]).to(self.device) # tokenized.input_ids.squeeze().to(self.device)
        prompt_str = self.tokenizer.decode(input_ids.tolist())
        input_ids = input_ids.unsqueeze(0)
        # print(prompt_str)
        
        # Generate the output
        N_prompt = input_ids.numel()

        # beams = [(input_ids, 0.0)]
        beams = input_ids # beams(initially 1) x seqlen
        scores = torch.tensor([0.0]).to(self.device)
        eos_id = self.tokenizer.eos_token_id
        eos = torch.tensor([eos_id]).to(self.device)
        choices = self.pixel_ids + [self.sep, eos_id]
        score_threshold = -5.0
        
        # bfs candidate search
        candidates = []
        for _ in range(512): # max generation token length
            next_beams = []
            next_scores = []
            with torch.no_grad():
                output = self.model(input_ids=beams)
            logits = output.logits[:,-1,:] # beams x vocab_size
            masked_logits = torch.full_like(logits, float('-inf'))
            masked_logits[:,choices] = logits[:,choices]
            probs = torch.softmax(masked_logits, dim=-1) # beams x vocab_size
            logp = torch.log(probs)
            for i in range(beams.shape[0]):
                valid = (logp[i]+scores[i]) >= score_threshold # find valid branches (boolean tensor of size vocab_size)
                if valid[eos_id]:
                    valid[eos_id] = False
                    beam_end = torch.cat([beams[i],eos], dim=0).detach().cpu() # add finished candidate
                    candidates.append(beam_end)
                next_tokens = torch.nonzero(valid) # valid_tokens x 1
                if next_tokens.size(0) == 0: break
                beam_copy = beams[i][None].expand(next_tokens.size(0),-1) # valid_tokens x seqlen
                next_beams.append(torch.cat([beam_copy, next_tokens], dim=1)) # append: valid_tokens x (seqlen+1)
                next_scores.append(logp[i][next_tokens]+scores[i]) # append: valid_tokens x 1
            if not next_beams: break
            beams = torch.cat(next_beams, dim=0) # new_beams x (seqlen+1)
            scores = torch.cat(next_scores, dim=0) # new_beams x 1
            # for beam in beams: print(self.tokenizer.batch_decode(beams))
            # print(scores)
        # print(candidates)

        # parse candidates into grid format
        grids = []
        for seq in candidates:
            seq = seq[N_prompt:].tolist()
            # print(self.tokenizer.decode(seq))
            try: # Parse the output
                grid = np.array(self.parse_grid(seq))
                # Inverse transformation
                inv_permu = np.empty_like(self.color_permu)
                inv_permu[self.color_permu] = np.arange(10)
                grid = inv_permu[grid]
                if self.flip != 2: 
                    grid = np.flip(grid, axis=self.flip)
                # print(grid, self.rot)
                grid = np.rot90(grid, -self.rot)
            except: grid = np.zeros((3,3))
            grids.append(grid)
            # print(grid)
        return grids


    def prepare_evaluation(self, select_adapter=None):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        adapter_path = ""
        self.prepare_train()
        if select_adapter is None:
            print("You did not define adapter path, so we use sample checkpoint instead.")
            adapter_path = "artifacts/checkpoint-final"
        else: adapter_path = "artifacts/" + select_adapter + "/checkpoint-final"
        self.model = PeftModel.from_pretrained(self.model, adapter_path) # use PeftModel class
        self.model.eval()

    def prepare_train(self):
        """ 
        If using an adapter, remove it from the model 
        (reverse of prepare_evaluation)
        """
        import gc
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True, # Allow the model to use custom code from the repository
            quantization_config=self.bnb_config, # Apply the 4-bit quantization configuration
            attn_implementation='sdpa', # Use scaled-dot product attention for better performance
            torch_dtype=torch.float16, # Set the data type for the model
            use_cache=False, # Disable caching to save memory
            device_map='auto', # {'': PartialState().process_index}, # map the model to available devices (e.g., GPUs)
            token=self.hf_token
        )

if __name__ == "__main__":
    solver = ARCSolver()

