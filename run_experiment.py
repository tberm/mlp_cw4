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
import torch.nn.functional as F
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

def run_prob_baseline(source, features, topic=None, save_to_file=True, plot_preds=False, note=None):

    start_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")

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
        if plot_preds:
            trues = scores[labels == 1]
            falses = scores[labels == 0]
            fig, ax = plt.subplots()
            ax.hist(falses, bins=300, alpha=0.5, label='false')
            ax.hist(trues, bins=300, alpha=0.5, label='true')
            ax.legend()
            fig.show()
        print(results)
    else:
        raise NotImplementedError("Haven't implemented training on multiple prob features")

    record_strings = [
        f'model={feature}_baseline',
        f'started_at={start_time}',
        f'val_source={source}',
    ] + [
        f'{key}={value}'
        for key, value in results.items()
    ]
    if note is not None:
        record_strings.append(f'note={note}')

    record_txt = '\n'.join(record_strings)
    print(record_txt)

    if save_to_file:
        folder = Path('probe-results')
        folder.mkdir(exist_ok=True)
        file_path = folder / f'{feature}_baseline_results_{start_time}.txt'
        with file_path.open('w', encoding='utf-8') as file:
            file.write(record_txt)

        print('Results saved to', str(file_path))



def run_experiment(train_data_args, val_data_args, model_name, learning_rates=[0.001],
    repeats=1, save_to_file=True, monitor_training=False, plot_preds=False, epochs=None,
    note=None):

    start_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    train_data = get_batch_of_embeddings(**train_data_args._asdict())
    val_data = get_batch_of_embeddings(**val_data_args._asdict())

    train_acts = torch.tensor(np.vstack(train_data['embeddings']), dtype=torch.float32)
    train_labels = torch.tensor(train_data['label'].to_numpy(), dtype=torch.float32)
    val_acts = torch.tensor(np.vstack(val_data['embeddings']), dtype=torch.float32)
    val_labels = torch.tensor(val_data['label'].to_numpy(), dtype=torch.float32)

    all_results = []
    for lr in learning_rates:
        for i in range(repeats):
            model = MODELS[model_name]()
            if monitor_training:
                monitor = TrainingMonitor(train_acts, train_labels,
                                          val_acts, val_labels)
                callback = monitor.update
            else:
                callback = None

            model.train_probe(train_acts=train_acts, train_labels=train_labels, 
                val_acts=val_acts, val_labels=val_labels, train_data_info=train_data_args,
                val_data_info=val_data_args, learning_rate=lr, epochs=epochs,
                training_epoch_callback=callback)

            val_probs = model.predict(val_acts)
            train_probs = model.predict(train_acts)
            results = evaluate_results(val_probs, val_labels, train_probs, train_labels)
            print(results)
            all_results.append(results)
            if plot_preds:
                #hist
                falses = val_probs[val_labels == 0]
                trues = val_probs[val_labels == 1]
                fig, ax = plt.subplots()
                ax.hist(falses, bins=100, alpha=0.5, label='false')
                ax.hist(trues, bins=100, alpha=0.5, label='true')
                ax.legend()
                fig.show()

                #lines
                fig2, ax2 = plt.subplots()
                color = ['green' if label else 'red' for label in val_labels]
                ax2.scatter([0]*len(val_probs), val_probs, color=color, marker='_', alpha=0.3, s=500)
                fig2.show()

            if monitor_training:
                monitor.plot_losses()

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
        f'epochs={epochs}',
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
    if note is not None:
        record_strings.append(f'note={note}')

    record_txt = '\n'.join(record_strings)
    print(record_txt)

    if save_to_file:
        folder = Path('probe-results')
        folder.mkdir(exist_ok=True)
        file_path = folder / f'{model_name}_baseline_results_{start_time}.txt'
        with file_path.open('w', encoding='utf-8') as file:
            file.write(record_txt)

        print('Results saved to', str(file_path))


class TrainingMonitor:
    def __init__(self, train_acts, train_labels, val_acts, val_labels):
        self.train_acts = train_acts
        self.train_labels = train_labels
        self.val_acts = val_acts
        self.val_labels = val_labels

        self.epochs = []
        self.train_accs = []
        self.val_accs = []
        self.cal_val_accs = []
        self.train_losses = []
        self.val_losses = []

    def update(self, model, epoch=None):
        if epoch is None:
            self.epochs.append(len(self.epochs) + 1)
        else:
            self.epochs.append(epoch)

        with torch.no_grad():
            train_probs = torch.tensor(model.predict(self.train_acts)).detach()
            val_probs = torch.tensor(model.predict(self.val_acts)).detach()
            #self.train_accs.append(accuracy_score(self.train_labels, train_probs.round()))
            self.train_losses.append(F.binary_cross_entropy(train_probs, self.train_labels))
            #self.val_accs.append(accuracy_score(self.val_labels, val_probs.round()))
            self.val_losses.append(F.binary_cross_entropy(val_probs, self.val_labels))
            #opt_thr = find_optimal_threshold(val_probs, self.val_labels)
            #self.cal_val_accs.append(accuracy_score(self.val_labels, val_probs > opt_thr))

    def plot_losses(self):
        fig, ax = plt.subplots()
        #ax.plot(self.epochs, self.train_accs, label='train acc')
        ax.plot(self.epochs, self.train_losses, label='train loss')
        #ax.plot(self.epochs, self.val_accs, label='val acc')
        ax.plot(self.epochs, self.val_losses, label='val loss')
        #ax.plot(self.epochs, self.cal_val_accs, label='calibrated val acc')
        ax.set_xlabel('Training epoch')
        ax.legend()
        fig.show()


def evaluate_results(val_probs, val_labels, train_probs=None, train_labels=None):
    out = {'accuracy': accuracy_score(val_labels, val_probs.round())}
    opt_thr = find_optimal_threshold(val_probs, val_labels)
    out['optimum_threshold'] = opt_thr
    out['calibrated_acc'] = accuracy_score(val_labels, val_probs > opt_thr)
    out['auroc'] = compute_roc_curve(val_labels, val_probs)[0]

    if train_probs is None and train_labels is None:
        return out

    opt_thr = find_optimal_threshold(train_probs, train_labels)
    out['optimum_threshold_on_train'] = opt_thr
    out['calibrated_on_train'] = accuracy_score(val_labels, val_probs > opt_thr)
    return out

def find_optimal_threshold(probs, labels):
    """
    Adapted from TrainSaplma.py
    """
    fpr_val, tpr_val, thresholds_val = roc_curve(labels, probs)
    optimal_threshold = thresholds_val[np.argmax([accuracy_score(labels, probs > thr) for thr in thresholds_val])]

    return optimal_threshold


def sanitise_nulls(value):
    if value == np.nan:
        return value
    if value == 'None':
        return np.nan
    return value


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
    df['train_layer'] = df.train_layer.astype(float)
    df['started_at'] = df.started_at.apply(
        lambda dt: datetime.strptime(dt, '%Y-%m-%d-%H%M%S')
    )
    # we have a mess of different null values for topic
    df['train_topic'] = df.train_topic.apply(sanitise_nulls)
    df['val_topic'] = df.val_topic.apply(sanitise_nulls)
    if sort_by is not None:
        return df.sort_values(sort_by)
    
    return df
