import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

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
        frame['source'] = source
        frames.append(frame[['statement', 'label', 'source']])

    df = pd.concat(frames)
    df = df.iloc[rng.permutation(len(df))]

    if number is not None:
        df = df.iloc[:number]

    if reindex:
        df = df.set_index([df.index, 'source'])

    return df


def get_batch_of_embeddings(number=None, split='train'):
    rng = np.random.default_rng(17)
    frames = []
    folder = Path('llama-true-false-results') / split
    for csv_path in folder.glob('*.csv'):
        frame = pd.read_csv(csv_path)
        source = str(csv_path).split('/')[-1].split('.')[0]
        frame['source'] = source
        frames.append(frame[['statement', 'label', 'source', 'embeddings']])

    df = pd.concat(frames)
    df = df.iloc[rng.permutation(len(df))]
 
    if number is None:
        return df

    return df.iloc[:number]