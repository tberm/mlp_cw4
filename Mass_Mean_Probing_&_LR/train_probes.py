import torch as t
from utils import DataManager
import random
from probes import LRProbe, MMProbe
import os

def main():
    # Hyperparameters
    model_size = '7B'
    split = 0.8
    seed = random.seed(10)

    # Setting up the device
    device = t.device("mps" if t.backends.mps.is_available() else "cpu")

    # Datasets
    #Each Medley is a list of training datasets. This allows to execute multiple experiments with same script
    train_medleys = [ 
        ['embeddings_neg_facts7B_4'],
    ]
    #Each Medley is evaluated on all validation datasets
    val_datasets = [
        'embeddings_neg_facts7B_4',
    ]

    # Probe Classes
    ProbeClasses = [LRProbe, MMProbe]

    # Initialize accuracies dictionary
    accs = {probe_class.__name__: {to_str(train_medley): {} for train_medley in train_medleys} for probe_class in ProbeClasses}

    # Create a directory for models if it does not exist
    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')

    # Loop over probes and datasets to train and evaluate
    for ProbeClass in ProbeClasses:
        for medley in train_medleys:
            # Set up data
            dm = DataManager()
            for dataset in medley:
                dm.add_dataset(dataset, split=split, seed=seed, center=True, device=device)
            for dataset in val_datasets:
                if dataset not in medley:
                    dm.add_dataset(dataset, split=None, center=True, device=device)

            # Train probe
            train_acts, train_labels = dm.get('train')
            probe = ProbeClass.from_data(train_acts, train_labels, device=device)

            #Save model
            model_path = f'trained_models/{ProbeClass.__name__}_{to_str(medley)}.pth'
            t.save(probe.state_dict(), model_path)

            # Evaluate
            for val_dataset in val_datasets:
                if val_dataset in medley:
                    acts, labels = dm.data['val'][val_dataset]
                    accs[ProbeClass.__name__][to_str(medley)][val_dataset] = (
                        probe.pred(acts, iid=False) == labels  # iid used to be true in the original code
                    ).float().mean().item()
                else:
                    acts, labels = dm.data[val_dataset]
                    accs[ProbeClass.__name__][to_str(medley)][val_dataset] = (
                        probe.pred(acts, iid=False) == labels
                    ).float().mean().item()

    # Save accuracies to file
    with open('validation_accuracies.txt', 'w') as f:
        f.write(f'Probe Class, Trained Medley Datasets, Validation Dataset: Accuracy \n')
        for probe_class, medleys in accs.items():
            for medley, datasets in medleys.items():
                for dataset, acc in datasets.items():
                    f.write(f'{probe_class}, {medley}, {dataset}: {acc}\n')

    print(accs)

def to_str(l):
    """Convert list to string."""
    return '+'.join(l)

if __name__ == "__main__":
    main()
