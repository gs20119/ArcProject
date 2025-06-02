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

class ARCSolver:
    """
    You should implement a `Solver` class for the project.
    """

    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct", token=None):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        self.config_path = "artifacts/config/config.yml"
        self.model_id = model_id
        self.token = token

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True, # Allow the model to use custom code from the repository
            quantization_config=self.bnb_config, # Apply the 4-bit quantization configuration
            attn_implementation='sdpa', # Use scaled-dot product attention for better performance
            torch_dtype=torch.float16, # Set the data type for the model
            use_cache=False, # Disable caching to save memory
            device_map='auto', # Automatically map the model to available devices (e.g., GPUs)
            token=token
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=token)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.system_header = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        self.user_header = "<|start_header_id|>user<|end_header_id|>\n"
        self.assist_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10) # "1" ~ "9"
            # self.tokenizer.encode(chr(i+65), add_special_tokens=False)[0] for i in range(10) # "A" ~ "J"
        ]
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def parse_grid(self, ids: List[int]):
        """
        Parse LLM generated sequence into ARC grid format.
        ids (List[int]): LLM generated token list 
        grid (List[List[int]]): parsed 2D grid
        """
        grid = []
        row = []
        inv_map = {k: i for i, k in enumerate(self.pixel_ids)}
        for idx in ids:
            if idx == self.sep:
                if len(row) > 0:
                    grid.append(row.copy())
                    row.clear()
            elif idx in inv_map: # skip spaces etc
                row.append(inv_map.get(idx))
        grid.append(row.copy())
        return grid

    def format_grid(self, grid: List[List[int]]):
        """
        Format 2D grid into LLM input tokens
        grid (List[List[int]]): 2D grid
        ids (List[int]): Parsed token list for LLM
        """
        ids = []
        for row in grid:
            for col in row:
                ids.append(self.pixel_ids[col])
            ids.append(self.sep)
        return ids

    def parse_grid_array(self, ids: List[int]):
        from ast import literal_eval
        text = self.tokenizer.decode(ids).split("\n")
        grid = None
        for line in text:
            if not line: continue
            if line[0] != "[": continue
            try: arr = literal_eval(line)
            except: continue
            grid = np.array(arr)
        return grid
    
    def text_grid_array(self, grid: List[List[int]]): # 2D array format, 1 line
        pixels = "["
        for row in grid:
            pixels += "["
            for col in row:
                pixels += str(col) + ", "
            if row is not None: pixels = pixels[:-2]
            pixels += "], "
        if grid is not None: pixels = pixels[:-2]
        pixels += "]\n"
        return pixels

    def parse_grid_coord(self, ids: List[int]):
        from ast import literal_eval
        self.color_map_inv = {
            "black": 0,
            "red": 1,
            "green": 2,
            "yellow": 3,
            "blue": 4,
            "magenta": 5,
            "cyan": 6,
            "white": 7,
            "purple": 8,
            "orange": 9,
        }
        text = self.tokenizer.decode(ids).split("\n")
        width, height = 1, 1
        grid = None
        for line in text[1:]:
            if ": " not in line: continue
            head, tail = line.split(": ")
            if head == "width": width = int(tail)
            if head == "height": height = int(tail)
            if head not in self.color_map_inv: continue
            if grid is None: grid = np.full((height, width), self.base_color).astype(int)
            try: coord = literal_eval(tail)
            except: continue
            for x, y in coord: 
                if 0 <= x and x < height and 0 <= y and y < width:
                    grid[x,y] = self.color_map_inv[head]
        return grid
    
    def text_grid_coord(self, grid: List[List[int]]):
        self.color_map = {
            0: "black",
            1: "red",
            2: "green",
            3: "yellow",
            4: "blue",
            5: "magenta",
            6: "cyan",
            7: "white",
            8: "purple",
            9: "orange",
        }
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
        text = "width: " + str(width) + "\nheight: " + str(height) + "\n"
        for i, values in coord_map.items():
            if not values: continue
            if i == self.base_color: continue
            text += self.color_map[i] + ": ["
            for coord in values:
                x, y = coord
                text += "(" + str(x) + "," + str(y) + "), "
            text = text[:-2]
            text += "]\n"
        return text
    
    def text_prompt_coord(self, datapoint, eval=False):
        """
        datapoint (dict): 
            contains training data, test input
        prompt (dict): string prompt
        """
        training_data = datapoint['train']
        input_test_data = datapoint['test'][0]['input']
        output_test_data = datapoint['test'][0]['output']

        sys = system_message_coord
        user = user_message_template1 + "\n"
        input_head = "input:\n"
        output_head = "output for the given input:\n"
        for ex in training_data:
            input = ex['input']
            output = ex['output']
            self.base_color = None
            user += input_head + self.text_grid_coord(input)
            user += output_head + self.text_grid_coord(output) + "\n"

        user += user_message_template2 + "\n"
        self.base_color = None
        user += input_head + self.text_grid_coord(input_test_data) + "\n"
        user += user_message_template3_coord + "\n"

        assist = output_head + self.text_grid_coord(output_test_data) + "\n"

        system_prompt = self.system_header + sys + "\n\n"
        user_prompt = self.user_header + user + "\n\n"
        assist_prompt = self.assist_header + assist
        prompt = system_prompt + user_prompt + assist_prompt
        
        return prompt

    def text_prompt_array(self, datapoint):
        """
        datapoint (dict): 
            contains training data, test input
        prompt (dict): string prompt
        """
        training_data = datapoint['train']
        input_test_data = datapoint['test'][0]['input']
        output_test_data = datapoint['test'][0]['output']

        sys = system_message_array
        user = user_message_template1 + "\n"
        input_head = "input:\n"
        output_head = "output for the given input:\n"
        for ex in training_data:
            input = ex['input']
            output = ex['output']
            self.base_color = None
            user += input_head + self.text_grid_array(input)
            user += output_head + self.text_grid_array(output) + "\n"

        user += user_message_template2 + "\n"
        self.base_color = None
        user += input_head + self.text_grid_array(input_test_data) + "\n"
        user += user_message_template3_array + "\n"

        assist = output_head + self.text_grid_array(output_test_data) + "\n"

        system_prompt = self.system_header + sys + "\n\n"
        user_prompt = self.user_header + user + "\n\n"
        assist_prompt = self.assist_header + assist
        prompt = system_prompt + user_prompt + assist_prompt
        return prompt
    

    def encode_prompt(self, datapoint):
        """
        datapoint (dict): 
            contains training data, test input
        prompt (dict): 
            dictionary that contains input ids and additional informations
        """
        training_data = datapoint['train']
        input_test_data = datapoint['test'][0]['input']

        sys = self.tokenizer.encode("<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + "\n" + system_message, add_special_tokens=False)
        user = self.tokenizer.encode("<|start_header_id|>user<|end_header_id|>" + "\n" + user_message_template1 + "\n", add_special_tokens=False)
        inp_desc = self.tokenizer.encode("input:\n", add_special_tokens=False)
        out_desc = self.tokenizer.encode("output:\n", add_special_tokens=False)
        for ex in training_data:
            inp = ex['input']
            out = ex['output']
            inp = self.format_grid(inp)
            out = self.format_grid(out)

            user += inp_desc
            user += inp
            user += out_desc
            user += out

        user += self.tokenizer.encode("\n" + user_message_template2 + "\n", add_special_tokens=False)

        user += inp_desc
        user += self.format_grid(input_test_data)
        user += self.tokenizer.encode("\n" + user_message_template3, add_special_tokens=False)

        messages = sys + user
        assis = self.tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)
        messages += assis

        return {
            "input_ids": messages,
            "input": input_test_data,
            "train": training_data
        }

    def preprocess_data(self, train_dataset):
        # generate and encode prompts 
        format_data = []
        for train_data in train_dataset:
            prompt = self.text_prompt_coord(train_data)
            tokenized = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding="max_length",
                max_length=4096,
                truncation=True
            )
            input_ids = tokenized.input_ids.squeeze()
            attn_mask = tokenized.attention_mask.squeeze()
            header_ids = self.tokenizer(self.assist_header, add_special_tokens=False)["input_ids"]

            input_ids_list = input_ids.tolist()
            header_len = len(header_ids)
            found = False
            for i in range(len(input_ids_list) - header_len+1):
                if input_ids_list[i:i+header_len] == header_ids:
                    label_idx = i + header_len
                    found = True
            if not found: assert False # If this error occurs, increase max_length
            
            labels = torch.full_like(input_ids, -100)
            labels[label_idx:] = input_ids[label_idx:]
            
            format_data.append({
                "input_ids": input_ids.tolist(),
                "attention_mask": attn_mask.tolist(),
                "labels": labels.tolist()
            })
        print(f"After truncating, there are {len(format_data)} rows left in the dataset.")
        dataset = Dataset.from_list(format_data)
        return dataset
        

    def train(self, train_dataset, prompt_mode='array', adapter_name=None, checkpoint=None):
        """
        Train a model with train_dataset. 
        Currently using QLoRA = Quantization + LoRA
        Read a project documentation for a description of `examples` and `question`.
        """
        # example
        datapoint = train_dataset[0]
        text_prompt = self.text_prompt_array if prompt_mode=='array' else self.text_prompt_coord
        prompt = text_prompt(datapoint)
        print(prompt)

        # generate and encode prompts 
        format_data = []
        for train_data in train_dataset:
            prompt = text_prompt(train_data)
            tokenized = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding="max_length",
                max_length=4096,
                truncation=True
            )
            input_ids = tokenized.input_ids.squeeze()
            attn_mask = tokenized.attention_mask.squeeze()
            header_ids = self.tokenizer(self.assist_header, add_special_tokens=False)["input_ids"]

            input_ids_list = input_ids.tolist()
            header_len = len(header_ids)
            found = False
            for i in range(len(input_ids_list) - header_len+1):
                if input_ids_list[i:i+header_len] == header_ids:
                    label_idx = i + header_len
                    found = True
            if not found: assert False # If this error occurs, increase max_length
            
            labels = torch.full_like(input_ids, -100)
            labels[label_idx:] = input_ids[label_idx:]
            
            format_data.append({
                "input_ids": input_ids.tolist(),
                "attention_mask": attn_mask.tolist(),
                "labels": labels.tolist()
            })
        
        print(f"After truncating, there are {len(format_data)} rows left in the dataset.")
        dataset = Dataset.from_list(format_data)
        # dataset = self.preprocess_data(train_dataset)
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "v_proj", "up_proj", "o_proj", "k_proj",
                "q_proj", "down_proj", "gate_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # create new LoRA finetuning adapter with adapter_name
        if adapter_name is None: 
            adapter_name = str(datetime.datetime.now()).split('.')[0].replace(":","").replace("-","").replace(" ","_")
        save_directory = "./artifacts/" + adapter_name
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)
        self.model.train()
        self.model.print_trainable_parameters()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        
        # continue training if checkpoint is provided TODO
        if checkpoint is not None: pass

        # Create Trainer and start training
        training_args = SFTConfig(
            output_dir=save_directory,
            per_device_train_batch_size=2, # set smaller when CUDA out of memory
            gradient_accumulation_steps=8, # set larger when CUDA out of memory
            gradient_checkpointing=True,
            num_train_epochs=2,
            learning_rate=2e-5,
            # lr_scheduler_type="cosine", # try using lr scheduler
            # warmup_steps=100,
            # weight_decay=0.01,
            fp16=True, # try using bf16
            save_strategy="epoch",
            logging_steps=10,
            remove_unused_columns=False,
            max_seq_length=2048
        )
        
        def data_collator(features):
            return {
                "input_ids": torch.tensor([ f["input_ids"] for f in features ]),
                "attention_mask": torch.tensor([ f["attention_mask"] for f in features ]),
                "labels": torch.tensor([ f["labels"] for f in features ])
            }
        
        # sample_data = dataset[0]
        # test_batch = {
        #     "input_ids": torch.tensor([sample_data["input_ids"]], device=self.device),
        #     "attention_mask": torch.tensor([sample_data["attention_mask"]], device=self.device),
        #     "labels": torch.tensor([sample_data["labels"]], device=self.device)
        # }
        # with torch.no_grad():
        #     outputs = self.model(**test_batch)
        # print(f"테스트 로스: {outputs.loss.item()}")

        trainer = SFTTrainer(
            model=self.model, 
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )

        # with torch.no_grad():
        #     batch = next(iter(trainer.get_train_dataloader()))
        #     outputs = self.model(**batch)
        #     print("outputs.loss:", outputs.loss)
        #     print("type:", outputs.loss.dtype)
        #     print("requires_grad:", getattr(outputs.loss, "requires_grad", None))
        #     print("grad_fn:", getattr(outputs.loss, "grad_fn", None))
        # assert False

        trainer.train()
        trainer.save_model(save_directory + "/checkpoint-final")




    def predict(self, datapoint, prompt_mode='array'):
        """
        A single example of test data is given.
        You should predict 2D grid (List[List[int]] or np.ndarray)
        """

        prompt = (self.text_prompt_array if prompt_mode=='array' else self.text_prompt_coord)(datapoint)
        tokenized = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding="max_length",
            max_length=2048,
            truncation=True
        )

        input_ids = tokenized.input_ids.squeeze().to(self.device)
        header_ids = self.tokenizer(self.assist_header, add_special_tokens=False)["input_ids"]
        header_str = self.tokenizer.decode(header_ids)
        prompt_str = self.tokenizer.decode(input_ids.tolist())
        print(prompt_str) # please check if max_length is appropriate.
        
        label_idx = prompt_str.find(header_str) + len(header_str)
        label_idx = len(self.tokenizer(prompt_str[:label_idx], add_special_tokens=False)["input_ids"])
        input_ids = input_ids[:label_idx].unsqueeze(0)

        # input_ids = torch.tensor(
        #     self.tokenizer(prompt),#prompt['input_ids'], 
        #     dtype=torch.long).to(self.device).view(1, -1)

        config = GenerationConfig(
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=512,
            # temperature=0.3,
            # top_k=5,
            # top_p=0.9
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
        parse_grid = self.parse_grid_array if prompt_mode=='array' else self.parse_grid_coord
        grid = parse_grid(output)
        # grid = np.array(self.parse_grid(output))
        # grid = grid[:x, :y]
        
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
            device_map='auto', # Automatically map the model to available devices (e.g., GPUs)
            token=self.token
        )
        # if hasattr(self.model, "peft_config") and self.model.peft_config:
        #     adapters = list(self.model.peft_config.keys())
        #     for name in adapters:
        #         self.model.delete_adapter(name)
        #     self.model = self.model.base_model
            
        

if __name__ == "__main__":
    solver = ARCSolver()




