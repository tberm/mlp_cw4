"""
Top level script for running our experiments with different probe models / datasets.
Easiest way is to start a python shell and run experiments interactively.

Specify the parameters for the datasets used for training and validation:

>>> from run_experiment import *
>>> train_dataset_args = DatasetArgs(source='tf', topic=None, layer=-1, split='train') 
>>> val_dataset_args = DatasetArgs(source='qa', topic=None, layer=-1, split='val') 

Run the experiment, specifying the method to use and how many times to repeat (results
will be averaged out):

>>> run_experiment(train_dataset_args, val_dataset_args, 'lr', repeats=5, monitor_training=True)

The results will be saved to a new file in probe-results.

You can use the `load_results` to get all the results stored in the probe-results folder
formatted as a DataFrame.


from run_experiment import *
train_dataset_args = DatasetArgs(source='tf', topic=None, layer=-1, split='train') 
val_dataset_args = DatasetArgs(source='tf', topic=None, layer=-1, split='val') 
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
run_experiment(train_dataset_args, val_dataset_args, 'lr', learning_rates, repeats=1, monitor_training=True)



"""
from collections import namedtuple
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, accuracy_score

from TrainSaplma import ProbeNetwork as SAPLMA
from TrainSaplma import compute_roc_curve
from mass_mean_and_lr.probes import LRProbeWrapper, MMProbeWrapper, LRProbe
from data_loader import get_batch_of_embeddings, get_prob_stats


DatasetArgs = namedtuple('DatasetArgs', 'source topic layer split')

MODELS = {
    'saplma': SAPLMA,
    'lr': LRProbeWrapper,
    'mm': MMProbeWrapper,
}

def run_prob_baseline(source, features, topic=None, save_to_file=True):
    data = get_prob_stats(source, topic)
    
    if isinstance(features, str):
        features = [features]

    if len(features) == 1:
        feature = features[0]
        labels = data['label'].to_numpy()
        scores = data[feature].to_numpy()
        # we expect *low* entropy to correlate with truth
        if 'entropy' in feature:
            scores = - scores
        results = evaluate_results(scores, labels)
        print(results)
    else:
        raise NotImplementedError("Haven't implemented training on multiple prob features")

def run_experiment(train_data_args, val_data_args, model_name, learning_rates=[0.001], repeats=1, save_to_file=True, monitor_training=False, plot_preds=False):

    start_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    
    train_data = get_batch_of_embeddings(**train_data_args._asdict())
    val_data = get_batch_of_embeddings(**val_data_args._asdict())

    train_acts = torch.tensor(np.vstack(train_data['embeddings']), dtype=torch.float32)
    train_labels = torch.tensor(train_data['label'].to_numpy(), dtype=torch.float32)
    val_acts = torch.tensor(np.vstack(val_data['embeddings']), dtype=torch.float32)
    if monitor_training==True and model_name=='lr':
        val_labels = torch.tensor(np.vstack(val_data['label']), dtype=torch.float32)
    else:
        val_labels = val_data['label'].to_numpy()


    all_results = []
    for lr in learning_rates:
        for i in range(repeats):
            model = MODELS[model_name]()
            if monitor_training and model_name == 'lr':
                model.train_probe(train_acts=train_acts, train_labels=train_labels, 
                                val_acts=val_acts, val_labels=val_labels, 
                                train_data_info=train_data_args, val_data_info=val_data_args, 
                                learning_rate=lr)
            else:
                model.train_probe(train_acts=train_acts, train_labels=train_labels, 
                                train_data_info=train_data_args, val_data_info=val_data_args, 
                                learning_rate=lr)

            probs = model.predict(val_acts)
            results = evaluate_results(probs, val_labels)
            print(results)
            all_results.append(results)
            if plot_preds:
                color = ['green' if label else 'red' for label in val_labels]
                plt.scatter([0]*len(probs), probs, color=color, marker='_', alpha=0.3, s=20)
                plt.show()

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


def load_results(sort_by=None):
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
    df = pd.DataFrame(rows)
    df['train_layer'] = df.train_layer.astype('int8')
    df['started_at'] = df.started_at.apply(
        lambda dt: datetime.strptime(dt, '%Y-%m-%d-%H%M%S')
    )
    if sort_by is not None:
        return df.sort_values(sort_by)
    
    return df