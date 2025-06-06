
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os, gc
import torch

from transformers import set_seed
from datasets import load_dataset
from evaluate import *
from arc.arc import ARCSolver

from datasets import Dataset
from utils import render_grid

# prepare the test dataset
data_path = "/workspace/dataset"
dataset, task_list = load_data(data_path)
df300 = sample_data(dataset, task_list, n_row=30000) 

# prepare samples for each task
task_samples = []
for t in range(300):
    df = sample_data(dataset, task_list, n_row=1000, indices=[t])
    task_samples.append(df)

simple_tasks = []
hard_tasks = []
for task_idx in range(300):
    check = True
    for data in Dataset.from_pandas(task_samples[task_idx]).shuffle().select(range(3)):
        for case in data['train']:
            wi, hi = len(case['input'][0]), len(case['input'])
            wo, ho = len(case['output'][0]), len(case['output'])
            if (wi!=wo) or (hi!=ho): check = False
        case = data['test'][0]
        wi, hi = len(case['input'][0]), len(case['input'])
        wo, ho = len(case['output'][0]), len(case['output'])
        if (wi!=wo) or (hi!=ho): check = False
    if check: simple_tasks.append(task_idx)
    else: hard_tasks.append(task_idx)
print(hard_tasks)

# load our model(arcsolver) instance
set_seed(1234567890)
os.environ['NCCL_P2P_DISABLE']='1'
os.environ['NCCL_IB_DISABLE']='1'
token = os.environ.get("HF_TOKEN", None)
solver = ARCSolver(model_id="Qwen/Qwen3-4B", hf_token=token)

# prepare train and then train
solver.prepare_train()
# n_train = len(hard_tasks)*700
n_train = len(simple_tasks)*500
n_eval = 500
dfsimple = sample_data(dataset, task_list, n_row=n_train+n_eval, indices=simple_tasks, random=56)
train_dataset = Dataset.from_pandas(dfsimple).select(range(n_train))
solver.train(train_dataset, checkpoint="20250606_053005/checkpoint-final")