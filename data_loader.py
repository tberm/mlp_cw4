import subprocess
from pathlib import Path
import re

import numpy as np
import pandas as pd

import os

def get_file_word_count(file):
    return int(subprocess.check_output(['wc', file]).decode().split()[0])


def get_batch_of_statements(number=None, split='train', reindex=False):
    """
    Return the required number of statement samples from our mixed datasets, randomly
    sampling to give a representative number of examples from each source.
    Or return all if number=None
    """
    rng = np.random.default_rng(17)
    frames = []
    folder = Path('mixed_true_false_data') / split
    for csv_path in folder.glob('*.csv'):
        frame = pd.read_csv(csv_path)
        source = str(csv_path).split('/')[-1].split('.')[0]
        frame['source'] = source #source is the dataset name, so like 'anims_true_false' or like 'numbers'
        frames.append(frame[['statement', 'label', 'source']])

    df = pd.concat(frames)
    df = df.iloc[rng.permutation(len(df))]

    if number is not None:
        df = df.iloc[:number] #get the first number of statements

    if reindex:
        df = df.set_index([df.index, 'source'])

    return df

def get_batch_of_embeddings(number=None, source='tf', split='train', layer=-1, topic=None):
    rng = np.random.default_rng(17)
    frames = []
    data_folder = 'llama-qa-results' if source == 'qa' else 'llama-true-false-results'
    path = Path(__file__).parent.resolve() / data_folder / split

    layer_pat = str(layer) if layer is not None else r'[\-0-9]+'
    if source == 'qa':
        pattern = 'answers_' + layer_pat + '.csv'
    else:
        pattern = 'embeddings(.*)7B_' + layer_pat + '.csv'

    pattern = re.compile(pattern)

    data_files = []
    for csv_path in path.glob('*.csv'):
        filename_match = pattern.match(csv_path.name)
        if filename_match is None:
            continue

        data_files.append(str(csv_path))
        frame = pd.read_csv(csv_path)
        if source == 'qa':
            frame.rename({'answer_is_correct': 'label'}, axis='columns', inplace=True)
            tpl = 'Q: {} A: {}'
            frame['statement'] = frame.apply(lambda row: tpl.format(row.question, row.answer), axis=1)
        topic = 'qa' if source == 'qa' else filename_match.groups()[0]
        frame['topic'] = topic
        frame['layer'] = int(layer)
        frame['label'] = frame.label.astype(int)
        frames.append(frame[['statement', 'label', 'topic', 'layer', 'embeddings']])

    print('Loaded data from:\n\n', '\n'.join(data_files))
    df = pd.concat(frames)
    df = df.iloc[rng.permutation(len(df))]
 
    if number is None:
        return df

    return df.iloc[:number]