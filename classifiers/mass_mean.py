import torch as t
from tqdm import tqdm
import os
from datetime import datetime


class MMProbeWrapper:
    def train_probe(self, train_acts, train_labels, *args, **kwargs):
        self.model = MMProbe.from_data(train_acts, train_labels)

    def predict(self, acts):
        with t.no_grad():
            return self.model(acts).numpy()


class MMProbe(t.nn.Module):
    def __init__(self, direction, atol=1e-3):
        super().__init__()
        self.direction = t.nn.Parameter(direction, requires_grad=False)


    def forward(self, x):
        return t.nn.Sigmoid()((x / t.norm(x)) @ self.direction)


    def pred(self, x):
        return self(x).round()

    def from_data(acts, labels, atol=1e-3, device='cpu'):
        acts, labels = acts.to(device).to(dtype=t.float32), labels.to(device).to(dtype=t.float32)
        pos_acts, neg_acts = acts[labels==1], acts[labels==0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean
        direction = direction / t.norm(direction)

        probe = MMProbe(direction).to(device)

        return probe
