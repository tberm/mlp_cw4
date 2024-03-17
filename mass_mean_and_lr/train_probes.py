import torch as t
from utils import DataManager
import random
from probes import LRProbe, MMProbe
import os
import sys
# Append the parent directory to sys.path to allow importing from it
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
from data_loader import get_batch_of_embeddings
from tqdm import tqdm

def main():
    # Hyperparameters
    model_size = '7B'
    seed = random.seed(17)

    # Setting up the device
    device = t.device("mps" if t.backends.mps.is_available() else "cpu")

    # Probe Classes
    ProbeClasses = [LRProbe, MMProbe]

    # Initialize accuracies dictionary
    accs = {probe_class.__name__: 0 for probe_class in ProbeClasses}

    # Create a directory for models if it does not exist
    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')

    train_data = get_batch_of_embeddings(split='train', layer=-1)
    val_data = get_batch_of_embeddings(split='val', layer=-1)
    train_acts = t.tensor(np.vstack(train_data['embeddings']), dtype=t.float32)
    train_labels = t.tensor(train_data['label'].to_numpy(), dtype=t.float32)
    val_acts = t.tensor(np.vstack(val_data['embeddings']), dtype=t.float32)
    val_labels = t.tensor(val_data['label'].to_numpy(), dtype=t.float32)

    # Loop over probes and datasets to val and evaluate
    for ProbeClass in tqdm(ProbeClasses, desc="Training Probes"):
        probe = ProbeClass.from_data(train_acts, train_labels, device=device)

        #Save model
        model_path = f'trained_models/{ProbeClass.__name__}.pth'
        t.save(probe.state_dict(), model_path)

        # Evaluate
        accs[ProbeClass.__name__]= (
            probe.pred(val_acts, iid=False) == val_labels
        ).float().mean().item()

    # Save accuracies to file
    with open('validation_accuracies-no-dm.txt', 'w') as f:
        f.write(f'Probe Class, Accuracy \n')
        for probe_class, acc in accs.items():
            f.write(f'{probe_class}: {acc}\n')

    print(accs)


if __name__ == "__main__":
    main()
