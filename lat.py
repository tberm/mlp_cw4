"""
Implementation of LAT from Zou et al (2023) https://arxiv.org/pdf/2310.01405.pdf

Test of how PC changes with more data:

    Getting PC for 5 pairs
    Getting PC for 10 pairs
    Dot prod with last PC is 0.6791404744838188
    Getting PC for 20 pairs
    Dot prod with last PC is 0.8782564310047014
    Getting PC for 40 pairs
    Dot prod with last PC is 0.9407143671935939
    Getting PC for 80 pairs
    Dot prod with last PC is 0.9508949596988319
    Getting PC for 160 pairs
    Dot prod with last PC is 0.9848249197052708
    Getting PC for 320 pairs
    Dot prod with last PC is 0.9879344522295015
    Getting PC for 640 pairs
    Dot prod with last PC is 0.9903457344444526
    Getting PC for 1280 pairs
    Dot prod with last PC is 0.9969290615682445
    Getting PC for 2560 pairs
    Dot prod with last PC is 0.9982798775964266
"""
from argparse import ArgumentParser
from ast import literal_eval
import csv
from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, GPTNeoXForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt

from mixed_true_false_data_loader import get_batch_of_statements


PROMPT_TEMPLATE = "\n".join([
    "Consider the amount of truthfulness in the following statement:",
    "Statement: {}",
    "The amount of truthfulness in the statement is",
])

RNG = np.random.default_rng()

def make_pairs(statements):
    """
    Given a DataFrame of statements and their true/false labels, arrange these into
    pairs where each pair consists of (true statement, false statement).

    We pair up sentences that come from the same data source, which should make it
    easier to isolate truthfulness as a difference
    """

    df = statements.reset_index()
    groups = dict(df.groupby(['source', 'label']).groups)
    paired_idxs = []

    for source in df.source.unique():
        idx_true = groups[(source, 1)]
        idx_false = groups[(source, 0)]
        min_len = min(len(idx_false), len(idx_true))
        idx_true = idx_true[:min_len]
        idx_false = idx_false[:min_len]
        paired_idxs.append(
            pd.DataFrame({'idx_true': idx_true, 'idx_false': idx_false})
        )

    paired_idx = pd.concat(paired_idxs)
    paired_idx = paired_idx.iloc[RNG.permutation(len(paired_idx))]
    for i, row in paired_idx.iterrows():
        yield (df.loc[row.idx_true], df.loc[row.idx_false])


def load_llama_model(model_path, cpu_only=False):
    model = LlamaForCausalLM.from_pretrained(model_path, local_files_only=True, device_map=None if cpu_only else 'auto')
    #tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def load_pythia_model(model_name):
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m",
        cache_dir="./pythia-70m/main",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m",
        cache_dir="./pythia-70m/main",
    )
    return model, tokenizer


class ResultsFile:
    def __init__(self, output_path, idx, layers):
        self.output_cols = [
            'true_statement_idx', 'false_statement_idx', 'source'
        ] + [
            f'rep_diff_layer_{layer}' for layer in layers
        ]
        self.output_file = output_path / f'rep_diffs-{idx}.csv'
        with self.output_file.open('w') as csv_file:
            writer = csv.DictWriter(csv_file, self.output_cols)
            writer.writeheader()

    def write_row(self, row):
        with self.output_file.open('a') as csv_file:
            writer = csv.DictWriter(csv_file, self.output_cols)
            writer.writerow(row)


def run_rep_extraction(model_name, output_path, cpu_only=False):

    #output_path.mkdir(exist_ok=False)

    #layers = [-1, -5, -9, -13]
    layers = [-1, -5]
    print(f'Running representation extraction for layers {layers}')
    rng = np.random.default_rng(17)
    statements = get_batch_of_statements()
    # shuffle because we may use only a subset
    statements = statements.iloc[rng.permutation(len(statements))]

    with open("config.json") as config_file:
        config = json.load(config_file)
    if 'llama' in model_name:
        llama_model_path = Path(config['llama_model_path'])
        model, tokenizer = load_llama_model(llama_model_path, cpu_only)
    elif 'pythia' in model_name:
        model, tokenizer = load_pythia_model(model_name)

    device = 'cpu' if cpu_only else 'cuda:0'

    # we'll save the diffs to checkpoint files of exponentially increasing size the idea
    # is to build the LAT probes on successive checkpoints and monitor to what extent
    # adding more data helps (they largest number of prompts they use in the paper is
    # 128)
    checkpoint_idx = 1
    output_file = ResultsFile(output_path, checkpoint_idx, layers)
    next_checkpoint_at = 5

    pairs = list(make_pairs(statements))

    pairs_progress = tqdm(pairs)
    for i, pair in enumerate(pairs_progress):
        if i == next_checkpoint_at:
            checkpoint_idx += 1
            output_file = ResultsFile(output_path, checkpoint_idx, layers)
            next_checkpoint_at = 2 * next_checkpoint_at

        pairs_progress.set_description(f'Working on checkpoint {checkpoint_idx}')

        prompts = [PROMPT_TEMPLATE.format(statement) for statement in pair]
        true_reps_by_layer = {}
        false_reps_by_layer = {}

        # get representations for true statement 
        inputs = tokenizer(prompts[0], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True) 
            for layer in layers:
                true_reps_by_layer[layer] = outputs.hidden_states[layer][0][-1].cpu().numpy()

        # get representations for false statement 
        inputs = tokenizer(prompts[1], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True) 
            for layer in layers:
                false_reps_by_layer[layer] = outputs.hidden_states[layer][0][-1].cpu().numpy()

        this_row = {
            'source': pair[0].source,
            'true_statement_idx': pair[0]['index'],
            'false_statement_idx': pair[1]['index'],
        }
        for layer in layers:
            col_name = f'rep_diff_layer_{layer}'
            diff = true_reps_by_layer[layer] - false_reps_by_layer[layer]
            this_row[col_name] = diff.tolist()

        output_file.write_row(this_row)

    print('Embedding diffs extracted for all training pairs.')


def normalise_vec(vec):
    return (vec - vec.mean()) / vec.std()


class LATProbe:
    def train(self, diffs, layer):
        self.layer = layer
        self.num_pairs = len(diffs)
        norm_diffs = diffs[f'rep_diff_layer_{layer}'].apply(literal_eval).apply(np.array).apply(normalise_vec)
        norm_diffs = np.array([emb for emb in norm_diffs])
        pca = PCA(1)
        pca.fit(norm_diffs)
        pc = pca.components_[0]
        self.pca = pca
        self.pc = pc
        return pc

    def train_and_visualise(self, diffs, layer):
        norm_diffs = diffs[f'rep_diff_layer_{layer}'].apply(literal_eval).apply(np.array).apply(normalise_vec)
        norm_diffs = np.array([emb for emb in norm_diffs])
        self.pca = PCA(2)
        proj = self.pca.fit_transform(norm_diffs)
        ...

    def __repr__(self):
        return f'<LATProbe trained on {self.num_pairs} pairs for layer {self.layer}>'


def get_validation_set(layer, number=500):
    """
    Problem: we generated the rep_diffs files before removing some items from the mixed
    t/f dataset. And to get a validation set we want to use the rep_diffs file to
    determine the indexes of examples not used in training, and then get embeddings for
    these from the general embeddings results used by other methods... but these might
    not match.
    We accept this and just ignore the validation set ids that don't have a
    corresponding entry in the embeddings results 
    """
    statements = get_batch_of_statements(reindex=True)
    val_ids = pd.read_csv(output_path / 'validation_set.csv') 
    embs_path = Path('llama-true-false-results')
    val_rows = []
    for i, row in val_ids.iterrows():
        topic_name = 'numbers' if row['source'] in ('larger_than', 'smaller_than') else row['source']
        emb_file = f'embeddings_{topic_name}7B_{layer}.csv'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('action', choices=['extract', 'train_probe'])
    parser.add_argument('-m', '--model')
    parser.add_argument('--diffs-path', help='path to folder of extracted representation differences', required=True)
    parser.add_argument('--cpu-only', action='store_true')

    args = parser.parse_args()
    diffs_path = Path(args.diffs_path)

    if args.action == 'extract':
        if args.model is None:
            raise ValueError('Must specify a `--model` for extract mode')
        run_rep_extraction(
            args.model, diffs_path, args.cpu_only
        )
    elif args.action == 'train_probe':
        diffs = None
        last_pc = None

        # use the last values of the last batch for validation since we don't need that
        # many values for training

        statements = get_batch_of_statements(reindex=True)
        validation_ids = pd.read_csv(diffs_path / 'validation_set.csv')

        for chunk in range(1, 11):
            new_diffs = pd.read_csv(diffs_path / f'rep_diffs-{chunk}.csv')
            if diffs is None:
                diffs = new_diffs
            else:
                diffs = pd.concat([diffs, new_diffs])

            print(f'Getting PC for {len(diffs)} pairs')
            pc = train_probe(diffs, -1)
            if last_pc is not None:
                print(f'Dot prod with last PC is {pc @ last_pc}')
            last_pc = pc
