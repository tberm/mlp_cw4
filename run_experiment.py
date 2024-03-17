"""
Top level script for running our experiments with different probe models / datasets.
Easiest way is to start a python shell and run experiments interactively.

Specify the parameters for the datasets used for training and validation:

>>> from run_experiment import *
>>> train_dataset_args = DatasetArgs(source='tf', topic=None, layer=-1, split='train') 
>>> val_dataset_args = DatasetArgs(source='qa', topic=None, layer=-1, split='val') 

Run the experiment, specifying the method to use and how many times to repeat (results
will be averaged out):

>>> run_experiment(train_dataset_args, val_dataset_args, 'saplma', repeats=5)

The results will be saved to a new file in probe-results.

You can use the `load_results` to get all the results stored in the probe-results folder
formatted as a DataFrame.
"""
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, accuracy_score

from TrainSaplma import ProbeNetwork as SAPLMA
from TrainSaplma import compute_roc_curve
from mass_mean_and_lr.probes import LRProbeWrapper, MMProbeWrapper
from data_loader import get_batch_of_embeddings


DatasetArgs = namedtuple('DatasetArgs', 'source topic layer split')

MODELS = {
    'saplma': SAPLMA,
    'lr': LRProbeWrapper,
    'mm': MMProbeWrapper,
}

def run_experiment(train_data_args, val_data_args, model_name, repeats=1, save_to_file=True):

    start_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    
    train_data = get_batch_of_embeddings(**train_data_args._asdict())
    val_data = get_batch_of_embeddings(**val_data_args._asdict())

    train_acts = torch.tensor(np.vstack(train_data['embeddings']), dtype=torch.float32)
    train_labels = torch.tensor(train_data['label'].to_numpy(), dtype=torch.float32)
    val_acts = torch.tensor(np.vstack(val_data['embeddings']), dtype=torch.float32)
    val_labels = val_data['label'].to_numpy()

    all_results = []
    for i in range(repeats):
        model = MODELS[model_name]()
        model.train_probe(train_acts, train_labels)
        probs = model.predict(val_acts)
        results = evaluate_results(probs, val_labels)
        print(results)
        all_results.append(results)

    if repeats == 1:
        results = all_results[0]
    else:
        results_frame = pd.DataFrame(all_results)
        results = {
            result_key: results_frame[result_key].mean()
            for result_key in results_frame
        }
        results.update({
            result_key + '_std': results_frame[result_key].std()
            for result_key in results_frame
        })

    record_strings = [
        f'model={model_name}',
        f'started_at={start_time}',
        f'repeats={repeats}',
    ] + [
        f'train_{param}={value}'
        for param, value in train_data_args._asdict().items()
    ] + [
        f'val_{param}={value}'
        for param, value in val_data_args._asdict().items()
    ] + [
        f'{key}={value}'
        for key, value in results.items()
    ]
    record_txt = '\n'.join(record_strings)
    print(record_txt)

    if save_to_file:
        folder = Path('probe-results')
        folder.mkdir(exist_ok=True)
        file_path = folder / f'{model_name}_results_{start_time}.txt'
        with file_path.open('w', encoding='utf-8') as file:
            file.write(record_txt)

        print('Results saved to', str(file_path))


def evaluate_results(probs, labels):
    out = {'accuracy': accuracy_score(labels, probs.round())}
    opt_thr = find_optimal_threshold(probs, labels)
    out['optimum_threshold'] = opt_thr
    out['calibrated_acc'] = accuracy_score(labels, probs > opt_thr)
    out['auroc'] = compute_roc_curve(labels, probs)[0]
    return out


def find_optimal_threshold(probs, labels):
    """
    Adapted from TrainSaplma.py
    """
    fpr_val, tpr_val, thresholds_val = roc_curve(labels, probs)
    optimal_threshold = thresholds_val[np.argmax([accuracy_score(labels, probs > thr) for thr in thresholds_val])]

    return optimal_threshold


def load_results():
    """
    load results of past experiments from files and return as a data
    frame
    """
    rows = []
    for file_path in Path('probe-results').glob('*results*.txt'):
        with file_path.open(encoding='utf-8') as file:
            rows.append({
                line.strip().split('=')[0]: line.strip().split('=')[1]
                for line in file.readlines()
            })
    return pd.DataFrame(rows)

