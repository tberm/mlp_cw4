import torch as t
import pandas as pd
import os
from glob import glob
import random
import numpy as np
import ast  # Use ast.literal_eval to safely evaluate strings containing Python literals

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTS_BATCH_SIZE = 25


class DataManager:
    """
    Class for storing activations and labels from datasets of statements.
    """
    def __init__(self):
        self.data = {
            'train' : {},
            'val' : {}
        } # dictionary of datasets
        self.proj = None # projection matrix for dimensionality reduction

    # Function to convert string representation of lists to numpy arrays
    def parse_embeddings(self,embedding_str):
        # Safely convert string to list
        embedding_list = ast.literal_eval(embedding_str)
        # Convert list to numpy array and return
        return np.array(embedding_list, dtype=np.float32)
    
    def add_dataset(self, dataset,dataset_use='train', split=None, seed=None, scale=False, device='cpu'):
        """
        Add a dataset to the DataManager.
        label : which column of the csv file to use as the labels.
        If split is not None, gives the train/val split proportion. Uses seed for reproducibility.
        """
        #acts = collect_acts(dataset_name, model_size, layer, center=center, scale=scale, device=device)
        #df = pd.read_csv(os.path.join(ROOT, '../processed_datasets', f'{dataset_name}.csv'))
        df = dataset
        labels = t.Tensor(df['label'].values).to(device).to(dtype=t.float32)
        acts = t.tensor(np.stack(df['embeddings'].apply(self.parse_embeddings).to_numpy()), dtype=t.float32).to(device)


        if split is None:
            self.data[dataset_use] = acts, labels

        if split is not None:
            assert 0 < split and split < 1
            if seed is None:
                seed = random.randint(0, 1000)
            t.manual_seed(seed)
            train = t.randperm(len(df)) < int(split * len(df))
            val = ~train
            self.data['train'][dataset_name] = acts[train], labels[train]
            self.data['val'][dataset_name] = acts[val], labels[val]

    def get(self, datasets):
        """
        Output the concatenated activations and labels for the specified datasets.
        datasets : can be 'all', 'train', 'val', a list of dataset names, or a single dataset name.
        If proj, projects the activations using the projection matrix.
        """
        if datasets == 'all':
            data_dict = self.data
        elif datasets == 'train':
            data_dict = self.data['train']
        elif datasets == 'val':
            data_dict = self.data['val']
        elif isinstance(datasets, list):
            data_dict = {}
            for dataset in datasets:
                if dataset[-6:] == ".train":
                    data_dict[dataset] = self.data['train'][dataset[:-6]]
                elif dataset[-4:] == ".val":
                    data_dict[dataset] = self.data['val'][dataset[:-4]]
                else:
                    data_dict[dataset] = self.data[dataset]
        elif isinstance(datasets, str):
            data_dict = {datasets : self.data[datasets]}
        else:
            raise ValueError(f"datasets must be 'all', 'train', 'val', a list of dataset names, or a single dataset name, not {datasets}")
        # if proj and self.proj is not None:
        #     acts = t.mm(acts, self.proj)
        all_acts, all_labels = [], []
        acts,labels=data_dict
        all_acts.append(acts), 
        all_labels.append(labels)
        return t.cat(all_acts, dim=0), t.cat(all_labels, dim=0)




