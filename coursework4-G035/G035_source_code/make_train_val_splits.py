from pathlib import Path
import numpy as np
import pandas as pd


TRAIN_VAL_RATIO = (0.85, 0.15)
TRAIN_VAL_TEST_RATIO = (0.65, 0.15, 0.2)


def split_gen_tqa_data(balance=False):
    """
    Don't actually make train/val/test splits, just split into files by layer
    and do balancing
    """
    rng = np.random.default_rng(17)

    common_cols = {
        'questions': ['question', 'question_idx'],
        'answers': ['question', 'question_idx', 'answer', 'answer_is_correct'],
    }
    prob_cols = {
        'questions': ['next_token_log_prob','next_token_entropy'],
        #'answers': ['all_log_probs', 'avg_log_prob', 'sum_log_prob', 'avg_entropy'],
        'answers': ['avg_log_prob', 'sum_log_prob'],
    }

    folder = Path('llama-qa-gen-results')
    #for file_type in ['answers', 'questions']:
    for file_type in ['answers']:
        file = folder / (file_type + '.csv')
        print(file)
        dataset = pd.read_csv(file)

        if file_type == 'answers' and balance:
            # balance out true vs false
            num_true = dataset.answer_is_correct.sum()
            num_false = len(dataset) - num_true
            assert num_true < num_false
            num_to_rm = num_false - num_true

            false_idxs = dataset[~dataset.answer_is_correct].index.to_numpy()
            rng.shuffle(false_idxs)
            idxs_to_rm = false_idxs[:num_to_rm]
            dataset.drop(idxs_to_rm, axis='index', inplace=True)

        for layer in [-1, -5, -9, -13]:
            emb_col_name = f'activations_layer_{layer}'
            subframe = dataset[common_cols[file_type] + [emb_col_name]].rename(
                {emb_col_name: 'embeddings'}, axis='columns')

            new_file = folder/(file.name.replace('.csv', f'_{layer}.csv'))
            subframe.to_csv(new_file)

        # also do a file with the other prob-related stats
        subframe = dataset[common_cols[file_type] + prob_cols[file_type]]
        new_file = folder/(file.name.replace('.csv', '_prob-stats.csv'))
        subframe.to_csv(new_file)



def split_tqa_data():
    rng = np.random.default_rng()

    common_cols = {
        'questions': ['question', 'question_idx'],
        'answers': ['question', 'question_idx', 'answer', 'answer_is_correct'],
    }
    prob_cols = {
        'questions': ['next_token_log_prob','next_token_entropy'],
        'answers': ['all_log_probs', 'avg_log_prob', 'sum_log_prob', 'avg_entropy'],
    }

    folder = Path('llama-qa-results-help-prompt')
    for file_type in ['answers', 'questions']:
        file = folder / (file_type + '.csv')
        print(file)
        dataset = pd.read_csv(file)

        if file_type == 'answers':
            # balance out true vs false
            num_true = dataset.answer_is_correct.sum()
            num_false = len(dataset) - num_true
            assert num_true < num_false
            num_to_rm = num_false - num_true

            false_idxs = dataset[~dataset.answer_is_correct].index.to_numpy()
            rng.shuffle(false_idxs)
            idxs_to_rm = false_idxs[:num_to_rm]
            dataset.drop(idxs_to_rm, axis='index', inplace=True)


        perm_idxs = rng.permutation(len(dataset))
        cutoff1 = int(len(dataset) * TRAIN_VAL_TEST_RATIO[0])
        cutoff2 = int(len(dataset) * (TRAIN_VAL_TEST_RATIO[0] + TRAIN_VAL_TEST_RATIO[1]))
        train_idxs = perm_idxs[:cutoff1]
        val_idxs = perm_idxs[cutoff1:cutoff2]
        test_idxs = perm_idxs[cutoff2:]

        for split, idxs in zip(
            ['train', 'test', 'val'], [train_idxs, val_idxs, test_idxs]
        ):

            split_frame = dataset.iloc[idxs]

            for layer in [-1, -5, -9, -13]:
                emb_col_name = f'activations_layer_{layer}'
                subframe = split_frame[common_cols[file_type] + [emb_col_name]].rename(
                    {emb_col_name: 'embeddings'}, axis='columns')

                new_file = folder/split/(file.name.replace('.csv', f'_{layer}.csv'))
                subframe.to_csv(new_file)

            # also do a file with the other prob-related stats
            subframe = split_frame[common_cols[file_type] + prob_cols[file_type]]
            new_file = folder/split/(file.name.replace('.csv', '_prob-stats.csv'))
            subframe.to_csv(new_file)


def split_true_false_data():
    rng = np.random.default_rng()

    for dataset_file in Path('mixed_true_false_data').glob('*.csv'):
        print(dataset_file)
        topic = str(dataset_file).split('/')[-1].split('.')[0]

        dataset = pd.read_csv(dataset_file)
        perm_idxs = rng.permutation(len(dataset))
        cutoff = int(len(dataset) * TRAIN_VAL_RATIO[0])
        train_idxs = perm_idxs[:cutoff]
        val_idxs = perm_idxs[cutoff:]
        train_df = dataset.iloc[train_idxs]
        val_df = dataset.iloc[val_idxs]
        train_df.to_csv(f'mixed_true_false_data/train/{topic}.csv')
        val_df.to_csv(f'mixed_true_false_data/val/{topic}.csv')

        print(train_df.label.mean())
        print(val_df.label.mean())

        for layer in [-1, -5, -9, -13]:
            emb_file = f'embeddings_{topic}7B_{layer}.csv'
            print(emb_file)

            embs = pd.read_csv('llama-true-false-results/' + emb_file)
            train_embs = embs.iloc[train_idxs]
            val_embs = embs.iloc[val_idxs]
            train_embs.to_csv(f'llama-true-false-results/train/' + emb_file)
            val_embs.to_csv(f'llama-true-false-results/val/' + emb_file)

            assert train_df.iloc[0]['statement'] == train_embs.iloc[0]['statement']
            assert train_df.iloc[-1]['statement'] == train_embs.iloc[-1]['statement']
            assert len(train_df) == len(train_embs)
            assert val_df.iloc[0]['statement'] == val_embs.iloc[0]['statement']
            assert val_df.iloc[-1]['statement'] == val_embs.iloc[-1]['statement']
            assert len(val_df) == len(val_embs)

            print(train_embs.label.mean())
            print(val_embs.label.mean())


if __name__ == '__main__':
    import sys
    dataset = sys.argv[1].strip().lower()
    if dataset in ('true_false', 'true-false'):
        split_true_false_data()
    elif dataset in ('tqa', 'truthfulqa'):
        split_tqa_data()
    elif dataset == 'gen-tqa':
        split_gen_tqa_data()
