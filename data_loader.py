from ast import literal_eval
import subprocess
from pathlib import Path
import re

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


def get_prob_stats(source, topic=None):
    # currently we only have these for QA and it only really makes sense for evaluation
    data_folders = {
        'qa': 'llama-qa-results',
        'qa-gen': 'llama-qa-gen-results',
        'qa-fixed-probs': 'llama-qa-results-fixed-probs',
        'qa-gen-prof': 'llama-qa-results-gen-prof-prompt',
    }
    data_folder = data_folders[source]

    base_path = Path(__file__).parent.resolve() / data_folder
    if 'gen' in source:
        # no splits for generated answers
        frame = pd.read_csv(base_path / 'answers_prob-stats.csv')
        frame = frame.rename({
            'answer_is_correct': 'label',
        }, axis='columns')
    else:
        # made a bit of a mess when making the splits since the train/val/test question
        # splits don't necessarily match up to the train/val/test answer splits
        # so combine all the questions and then join to answers from the split we want
        train_qs = pd.read_csv(base_path / 'train' / 'questions_prob-stats.csv')
        val_qs = pd.read_csv(base_path / 'val' / 'questions_prob-stats.csv')
        test_qs = pd.read_csv(base_path / 'test' / 'questions_prob-stats.csv')
        questions = pd.concat([train_qs, val_qs, test_qs])
        answers = pd.read_csv(base_path / 'val' / 'answers_prob-stats.csv')

        questions = questions.set_index('question_idx')
        # remove columns clashing with answers
        questions = questions[['next_token_log_prob', 'next_token_entropy']]

        frame = answers.join(questions, on='question_idx')

        frame = frame.rename({
            'next_token_log_prob': 'answer_start_token_log_prob',
            'next_token_entropy': 'answer_start_token_entropy',
            'answer_is_correct': 'label',
        }, axis='columns')

    frame['label'] = frame.label.astype(int)
    return frame


def get_batch_of_embeddings(number=None, source='tf', split='train', layer=-1, topic=None):

    if topic is not None:
        filter_topics = [topic] if isinstance(topic, str) else topic

    rng = np.random.default_rng(17)
    frames = []
    data_folders = {
        'qa': 'llama-qa-results',
        'qa-fixed-probs': 'llama-qa-results-fixed-probs',
        'tf': 'llama-true-false-results',
        'qa-gen': 'llama-qa-gen-results',
        'qa-gen-bal': 'llama-qa-gen-results',
        'qa-gen-prof': 'llama-qa-results-gen-prof-prompt',
        'qa-gen-prof-bal': 'llama-qa-results-gen-prof-prompt',
    }
    data_folder = data_folders[source]
    if 'gen' in source:
        # we have no train/val splits for generated answers:
        paths = [Path(__file__).parent.resolve() / data_folder]
    elif split is None:
        paths = [
            Path(__file__).parent.resolve() / data_folder / split 
            for split in ['train', 'val', 'test']
        ]
    elif isinstance(split, str):
        paths = [Path(__file__).parent.resolve() / data_folder / split]
    else:
        # it's an array
        paths = [Path(__file__).parent.resolve() / data_folder / s for s in split]

    layer_pat = str(layer) if layer is not None else r'[\-0-9]+'

    if 'qa' in source:
        pattern = f'answers_{layer_pat}.csv'
    elif topic is not None:
        topics_pat = '(' + '|'.join(filter_topics) + ')'
        pattern = f'embeddings_{topics_pat}7B_{layer_pat}.csv'
    else:
        pattern = f'embeddings_(.*)7B_{layer_pat}.csv'

    pattern = re.compile(pattern)

    data_files = []
    for path in paths:
        for csv_path in path.glob('*.csv'):
            filename_match = pattern.match(csv_path.name)
            if filename_match is None:
                continue

            data_files.append(str(csv_path))
            frame = pd.read_csv(csv_path)
            if 'qa' in source:
                if 'bal' in source:
                    # balance true vs false in the dataset
                    surplus = frame.answer_is_correct.sum() - (~frame.answer_is_correct).sum()
                    assert surplus > 0, 'Should have more correct than incorrect'
                    true_idxs = frame[frame.answer_is_correct].index.to_numpy()
                    rng.shuffle(true_idxs)
                    idxs_to_rm = true_idxs[:surplus]
                    frame.drop(idxs_to_rm, axis='index', inplace=True)

                # Load the TruthfulQA CSV
                truthfulqa_path = Path(__file__).parent.resolve() / 'TruthfulQA' / 'TruthfulQA.csv'
                truthfulqa_df = pd.read_csv(truthfulqa_path)

                # Perform the join
                frame = pd.merge(
                    frame, truthfulqa_df[['Question', 'Category']],
                    left_on='question', right_on='Question', how='left'
                )

                # Filter frame_with_category to only include rows with 'Category' in the 'topic' list
                if topic is not None: 
                    frame = frame[frame['Category'].isin(filter_topics)]

                frame.rename({'answer_is_correct': 'label'}, axis='columns', inplace=True)
                frame.rename({'Category': 'topic'}, axis='columns', inplace=True)
                tpl = 'Q: {} A: {}'
                frame['statement'] = frame.apply(lambda row: tpl.format(row.question, row.answer), axis=1)
            else:
                frame['topic'] = filename_match.groups()[0]

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