import pandas as pd



def truncate_numbers_dataset():
    larger = pd.read_csv('geometry-of-truth/datasets/larger_than.csv')
    smaller = pd.read_csv('geometry-of-truth/datasets/smaller_than.csv')

    truncated = []
    for df in (larger, smaller):
        trues = df[df.label == 1].reset_index().rename({'index': 'original_idx'}, axis='columns')
        falses = df[df.label == 0].reset_index().rename({'index': 'original_idx'}, axis='columns')
        trunc_trues = trues[trues.index % 4 == 0]
        trunc_falses = falses[falses.index % 4 == 0]
        truncated += [trunc_trues, trunc_falses]

    final = pd.concat(truncated)
    print(f"New dataset has {len(final)} rows")
    final_trues = final[final.label == 1]
    print(f"{len(final_trues)} are true, with mean diff {final_trues['diff'].mean()}")
    final_falses = final[final.label == 1]
    print(f"{len(final_falses)} are false, with mean diff {final_falses['diff'].mean()}")

    final.to_csv('numbers.csv')


def truncate_inventions_dataset():
    inv = pd.read_csv('azaria-dataset/inventions_true_false.csv')
    trues = inv[inv.label == 1]
    falses = inv[inv.label == 0]
    new_len = min(len(trues), len(falses))
    trunc_trues = trues.sample(new_len)
    trunc_falses = falses.sample(new_len)
    final = pd.concat([trunc_trues, trunc_falses])

    print(f"New dataset has {len(final)} rows")
    final_trues = final[final.label == 1]
    print(f"{len(final_trues)} are true")
    final_falses = final[final.label == 1]
    print(f"{len(final_falses)} are false")

    final.to_csv('inventions_true_false_balanced.csv')




