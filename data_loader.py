from ast import literal_eval
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


def parse_embedding(embedding_str):
    """
    much faster str->arr conversion than ast.literal_eval
    """
    return np.fromstring(embedding_str[1:-1], sep=',', dtype=np.float32)


def get_prob_stats(topic=None):
    # currently we only have these for QA and it only really makes sense for evaluation
    data_folder = 'llama-qa-results'

    # made a bit of a mess when making the splits since the train/val/test question
    # splits don't necessarily match up to the train/val/test answer splits
    # so combine all the questions and then join to answers from the split we want
    base_path = Path(__file__).parent.resolve() / data_folder
    train_qs = pd.read_csv(base_path / 'train' / 'questions_prob-stats.csv')
    val_qs = pd.read_csv(base_path / 'val' / 'questions_prob-stats.csv')
    test_qs = pd.read_csv(base_path / 'test' / 'questions_prob-stats.csv')
    questions = pd.concat([train_qs, val_qs, test_qs])
    questions = questions.set_index('question_idx')
    # remove columns clashing with answers
    questions = questions[['next_token_log_prob', 'next_token_entropy']]

    split = 'val'
    answers = pd.read_csv(base_path / split / 'answers_prob-stats.csv')

    frame = answers.join(questions, on='question_idx')

    frame = frame.rename({
        'next_token_log_prob': 'answer_start_token_log_prob',
        'next_token_entropy': 'answer_start_token_entropy',
        'answer_is_correct': 'label',
    }, axis='columns')

    frame['label'] = frame.label.astype(int)
    return frame


def get_batch_of_embeddings(number=None, source='tf', split='train', layer=-1, topic=None):
    if source not in ['tf', 'qa']:
        raise ValueError(f'Invalid `source`: {source}')
    if topic is not None and source != 'tf':
        raise ValueError(f'`topic` is only a valid parameter when `source="tf"`')

    rng = np.random.default_rng(17)
    frames = []
    data_folder = 'llama-qa-results' if source == 'qa' else 'llama-true-false-results'
    path = Path(__file__).parent.resolve() / data_folder / split

    layer_pat = str(layer) if layer is not None else r'[\-0-9]+'

    if source == 'qa':
        pattern = f'answers_{layer_pat}.csv'
    elif topic is not None:
        pattern = f'embeddings_{topic}7B_{layer_pat}.csv'
    else:
        pattern = f'embeddings_(.*)7B_{layer_pat}.csv'

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
            this_topic = 'qa'
        elif topic is None:
            this_topic = filename_match.groups()[0]
        else:
            this_topic = topic

        frame['topic'] = this_topic
        frame['layer'] = int(layer)
        frame['label'] = frame.label.astype(int)
        frames.append(frame[['statement', 'label', 'topic', 'layer', 'embeddings']])

    print('Loading data from:\n\n', '\n'.join(data_files))
    df = pd.concat(frames)
    df = df.iloc[rng.permutation(len(df))]
    df['embeddings'] = df.embeddings.apply(parse_embedding)
 
    if number is None:
        return df

    return df.iloc[:number]