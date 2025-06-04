import numpy as np
from tqdm.auto import tqdm
import os

from transformers import set_seed
from datasets import load_dataset
from statistics import quantiles, mean
import pandas as pd
import json

# compare
def check_match(pred, truth):
    pred = np.array(pred, dtype=np.uint8)
    truth = np.array(truth, dtype=np.uint8)
    if len(pred.shape) != 2 or pred.shape != truth.shape: return 0
    else: return int(np.all(pred == truth))


# prepare 
def load_data(base_dir):
    ########################################################################
    # load the dataset (json)
    # each element of the dataset is a list of cases in a single task
    # each sample is in a form of dictionary
    ########################################################################
    filenames = os.listdir(base_dir)
    data_files = [os.path.join(base_dir, p) for p in filenames if ".json" in p]
    dataset = []
    n_case = []
    for fn in data_files:
        with open(fn) as fp:
            data = json.load(fp)
        dataset.append(data) 
        n_case.append(len(data))
    Q = quantiles(n_case, n=4)
    task_names = [fn.split(".")[0] for fn in filenames]
    
    print(f"Imported {len(dataset)} different tasks in the dataset.")
    print(f"# of samples in a task: min({min(n_case)}), Q1({int(Q[0])}), Q2({int(Q[1])}), Q3({int(Q[2])}), max({max(n_case)}), mean({mean(n_case):.1f})")
    return dataset, task_names


def sample_data(dataset, tasks, n_row=10000, n_example=3, indices=list(range(300)), random=42):
    #####################################################################
    # each element of the dataset is a list of cases in a single task
    # n_row: num of total rows. MAX ~ 260000
    # indices: list of indices of selected tasks. subset of range(300)
    #####################################################################
    rng = np.random.default_rng(random)
    N_DATA, N_TASK = n_row, len(indices)

    data = []
    while len(data) < N_DATA:
        task_idx = rng.choice(indices, replace=True)
        task = dataset[task_idx]
        file_name = tasks[task_idx]

        # For each row, sample 4 cases from a task 
        # 3 of them are for train, and the last one is for test
        n_sample = len(task)
        grids_idx =  rng.choice(n_sample, size=n_example+1, replace=True) 
        train_grids = [task[i] for i in grids_idx[:n_example]]
        test_grids = [task[i] for i in grids_idx[n_example:]]

        test_inputs = [{'input': grid['input']} for grid in test_grids]
        test_outputs = [grid['output'] for grid in test_grids]
        test_outputs_transformed = [{'output': grid} for grid in test_outputs]
        combined_tests = []
        for test_input, test_output in zip(test_inputs, test_outputs_transformed):
            combined_tests.append({'input': test_input['input'], 'output': test_output['output']})

        data.append({
            'task': file_name,
            'train': train_grids,
            'test_input': test_inputs,
            'test_output': test_outputs,
            'test': combined_tests,
        })

    # Save the data as DataFrame
    df = pd.DataFrame(data)
    return df
