from pathlib import Path
import numpy as np
import pandas as pd


TRAIN_RATIO = 0.85


if __name__ == '__main__':

    rng = np.random.default_rng()

    for dataset_file in Path('mixed_true_false_data').glob('*.csv'):
        print(dataset_file)
        topic = str(dataset_file).split('/')[-1].split('.')[0]

        dataset = pd.read_csv(dataset_file)
        perm_idxs = rng.permutation(len(dataset))
        cutoff = int(len(dataset) * TRAIN_RATIO)
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

