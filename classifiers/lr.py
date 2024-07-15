import os
import torch as t
from tqdm import tqdm


class LRProbe(t.nn.Module):
    def __init__(self, d_in=4096):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, 1, bias=False),
            t.nn.Sigmoid()
        )

    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)

    def predict(self, acts):
        with t.no_grad():
            return self(acts).numpy()

    @classmethod
    def from_data(cls, train_acts, train_labels, lr=0.001, weight_decay=0.1,
                  epochs=None, device='cpu', val_acts=None, val_labels=None, train_data_info=None,
                  val_data_info=None, training_epoch_callback=None):
        train_acts, train_labels = train_acts.to(device), train_labels.to(device)
        if val_acts is not None and val_labels is not None:
            val_acts, val_labels = val_acts.to(device), val_labels.to(device)

        # Directory setup
        if train_data_info is None and val_data_info is None:
            today_str = datetime.today().strftime('%Y-%m-%d')
            results_dir = os.path.join('probe_results', 'lr_training_results', f'experiment_{today_str}')
            os.makedirs(results_dir, exist_ok=True)
        else:
            name = f"{lr}_{train_data_info.source}_{train_data_info.topic}_{train_data_info.layer}__to__{val_data_info.source}_{val_data_info.topic}_{val_data_info.layer}_{val_data_info.split}_{datetime.today().strftime('%Y-%m-%d')}"
            results_dir = os.path.join('probe-results', 'lr_training_results', name)
            os.makedirs(results_dir, exist_ok=True)

        probe = cls(train_acts.shape[-1]).to(device)

        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

        epochs = 1000 if epochs is None else epochs

        with open(os.path.join(results_dir, 'training_accuracy_per_epoch.txt'), 'w') as train_f, open(os.path.join(results_dir, 'validation_accuracy_per_epoch.txt'), 'w') as val_f:
            for epoch in tqdm(range(epochs), desc='Training LRProbe'):
                opt.zero_grad()
                outputs = probe(train_acts)
                loss = t.nn.BCELoss()(outputs, train_labels)
                loss.backward()
                opt.step()

                if training_epoch_callback is not None and epochs % 10 == 0:
                    training_epoch_callback(probe, epoch)

                # Calculate training accuracy
                preds = outputs.round()
                correct = (preds == train_labels).float().sum()
                accuracy = correct / train_labels.shape[0]
                train_f.write(f'Epoch {epoch+1}: {accuracy.item()}\n')

                # Calculate validation accuracy if validation data is provided
                if val_acts is not None and val_labels is not None:
                    with t.no_grad():
                        val_outputs = probe(val_acts)
                        val_preds = val_outputs.round()
                        #print("val_preds shape",val_preds.shape)
                        #print("val_labels shape",val_labels.shape)
                        val_labels_squeezed = val_labels.squeeze(-1)
                        correct_val = (val_preds == val_labels_squeezed).float().sum()
                        val_accuracy = correct_val / val_labels.shape[0]
                        val_f.write(f'Epoch {epoch+1}: {val_accuracy.item()}\n')

        return probe

    @property
    def direction(self):
        return self.net[0].weight.data[0]

class LRProbeWrapper:
    def train_probe(self, train_acts, train_labels, val_acts=None, val_labels=None,
                    train_data_info=None, val_data_info=None, learning_rate=0.001,
                    epochs=None, training_epoch_callback=None, **kwargs):
        self.model = LRProbe.from_data(train_acts=train_acts, train_labels=train_labels,
            val_acts=val_acts, val_labels=val_labels, train_data_info=train_data_info,
            val_data_info=val_data_info, lr=learning_rate, epochs=None,
            training_epoch_callback=training_epoch_callback)

    def predict(self, acts):
        with t.no_grad():
            return self.model(acts).numpy()


