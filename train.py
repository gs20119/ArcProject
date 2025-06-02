
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os, gc
import torch

from transformers import set_seed
from datasets import load_dataset
from evaluate import *
from arc import ARCSolver

from datasets import Dataset
from utils import render_grid

# prepare the test dataset
data_path = "dataset"
dataset, task_list = load_data(data_path)
df = sample_data(dataset, task_list, n_row=10000) 

# prepare samples for each task
task_samples = []
for t in range(300):
    df = sample_data(dataset, task_list, n_row=1000, indices=[t])
    task_samples.append(df)

# load our model(arcsolver) instance
set_seed(1234567890)
token = os.environ.get("HF_TOKEN", None)
solver = ARCSolver(model_id="Qwen/Qwen3-4B", token=token)

# prepare train and then train
solver.prepare_train()
n_train = 10000
df20 = sample_data(dataset, task_list, n_row=10500, indices=list(range(20)))
train_dataset = Dataset.from_pandas(df20).select(range(n_train))
solver.train(train_dataset)
