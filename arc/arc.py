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

from accelerate import Accelerator, PartialState, InitProcessGroupKwargs
from transformers import modeling_utils

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
        self.accelerator = Accelerator(kwargs_handlers = [InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))])
        if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
            modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

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

        self.color_map = {
            0: "black", 1: "red", 2: "green", 3: "yellow", 4: "blue",
            5: "magenta", 6: "cyan", 7: "white", 8: "purple", 9: "orange",
        }
        self.color_map_inv = {
            "black": 0, "red": 1, "green": 2, "yellow": 3, "blue": 4,
            "magenta": 5, "cyan": 6, "white": 7, "purple": 8, "orange": 9,
        }

        self.system_header = "<|im_start|>system\n"
        self.user_header = "<|im_start|>user\n"
        self.assist_header = "<|im_start|>assistant\n"
        
        self.rng = np.random.default_rng(24)
        
        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def parse_grid(self, ids: List[int]):
        from ast import literal_eval
        text = self.tokenizer.decode(ids).split("\n")
        
        for line in text[::-1]:
            if "rest" not in line: continue
            words = line.split(" ")
            if words[1] in self.color_map_inv:
                self.base_color = self.color_map_inv[words[1]]
            break

        width, height = 1, 1
        grid = None
        for line in text[1:-1]:
            if " " not in line: continue
            words = line.split(" ")
            if "width" in line and "height" in line:
                width = int(words[6][:-1])
                height = int(words[11][:-1])
            if "[" not in line: continue
            if grid is None: 
                grid = np.full((height, width), self.base_color).astype(int)
            count = int(words[0])
            color = words[1]
            if color not in self.color_map_inv: continue 
            try: 
                coord = ' '.join(words[5:5+count])
                coord = literal_eval(coord[:-1])
            except: continue
            for x, y in coord: 
                if 0 < x and x <= height and 0 < y and y <= width:
                    grid[x-1,y-1] = self.color_map_inv[color]
        
        if grid is None:
            grid = np.full((height, width), self.base_color).astype(int)
        return grid
    
    def text_grid(self, grid: np.ndarray):
        coord_map = dict()
        for c in range(10): coord_map[c] = []
        height = len(grid)
        for i, row in enumerate(grid):
            width = len(row)
            for j, col in enumerate(row):
                coord_map[col].append((i,j))
        if self.base_color is None:
            col_count = [ len(values) for i, values in coord_map.items() ]
            self.base_color = np.argmax(col_count)
        
        text = f"The grid has a width of {width}, and a height of {height}.\n"
        text += f"So, there are total {width*height} pixels in the grid, colored:\n"
        # text = "width: " + str(width) + "\nheight: " + str(height) + "\n"
        for i, values in coord_map.items():
            if not values: continue
            if i == self.base_color: continue
            count = len(values)
            # text += self.color_map[i] + ": ["
            text += f"{count} {self.color_map[i]} pixels on coordinates ["
            for coord in values:
                x, y = coord
                text += "(" + str(x+1) + "," + str(y+1) + "), "
            text = text[:-2]
            # text += "]\n"
            text += f"],\n"
        text += f"{len(coord_map[self.base_color])} {self.color_map[self.base_color]} pixels on the rest.\n"
        return text
    
    def text_prompt(self, datapoint, train=False, reasoning=False):
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

        sys = system_message
        user = user_message_template1 + "\n"
        input_head = "input:\n"
        output_head = "output for the given input:\n"
        for ex in training_data:
            input = transform(ex['input'])
            output = transform(ex['output'])          
            self.base_color = None
            user += input_head + self.text_grid(input)
            user += output_head + self.text_grid(output) + "\n"

        test_data = datapoint['test'][0]
        input_test_data = transform(test_data['input'])
        output_test_data = transform(test_data['output'])
        user += user_message_template2 + "\n"
        self.base_color = None
        user += input_head + self.text_grid(input_test_data) + "\n"
        user += user_message_template3 + "\n"

        assist = output_head + self.text_grid(output_test_data) + "\n"

        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assist}
        ]
        if not train: messages = messages[:2]
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=(not train),
            enable_thinking=reasoning
        )
        if not reasoning:
            prompt = prompt.replace("<think>\n\n","").replace("</think>\n\n","")
        return prompt


    def preprocess_data(self, train_dataset):
        # generate and encode prompts 
        from statistics import quantiles, mean
        format_data = []
        token_lens = []
        for train_data in train_dataset:
            prompt = self.text_prompt(train_data, train=True)
            tokenized = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding="max_length",
                max_length=2048,
                truncation=True
            )
            input_ids = tokenized.input_ids.squeeze()
            attn_mask = tokenized.attention_mask.squeeze()
            header_ids = self.tokenizer(self.assist_header, add_special_tokens=False)["input_ids"]
            # print(self.tokenizer.decode(tokenized["input_ids"][0]))
            # print(self.tokenizer.decode(header_ids))

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
            if len(format_data) % 1000 == 0: print(len(format_data))
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
        prompt = self.text_prompt(datapoint, train=True)
        print(prompt)

        # generate and encode prompts 
        dataset = self.preprocess_data(train_dataset)
        
        # create new LoRA finetuning adapter with adapter_name
        lora_config = LoraConfig(
            r=128, # 8 or 16
            lora_alpha=32, # 16 or 32
            # target_modules="all-linear",
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

        # self.model.to(self.device)
        self.accelerator.prepare(self.model)
        self.model.train()
        self.model.print_trainable_parameters()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        # Create Trainer and start training
        training_args = SFTConfig(
            output_dir=save_directory,
            per_device_train_batch_size=12, # set smaller when CUDA out of memory
            gradient_accumulation_steps=3, # set larger when CUDA out of memory
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
            logging_steps=10,
            remove_unused_columns=False,
            max_seq_length=2048,
            ddp_find_unused_parameters=True,
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
        prompt = self.text_prompt(datapoint)
        tokenized = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding="max_length",
            max_length=4096,
            truncation=True
        )

        input_ids = tokenized.input_ids.squeeze().to(self.device)
        header_ids = self.tokenizer(self.assist_header, add_special_tokens=False)["input_ids"]
        header_str = self.tokenizer.decode(header_ids)
        prompt_str = self.tokenizer.decode(input_ids.tolist())
        # print(prompt_str) # please check if max_length is appropriate.
        
        label_idx = prompt_str.find(header_str) + len(header_str)
        label_idx = len(self.tokenizer(prompt_str[:label_idx], add_special_tokens=False)["input_ids"])
        input_ids = input_ids[:label_idx].unsqueeze(0)

        config = GenerationConfig(
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=512,
            # temperature=0.6, # Qwen default setting
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
            grid = self.parse_grid(output)
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
            device_map={'': PartialState().process_index}, # 'auto', # map the model to available devices (e.g., GPUs)
            token=self.hf_token
        )

if __name__ == "__main__":
    solver = ARCSolver()

