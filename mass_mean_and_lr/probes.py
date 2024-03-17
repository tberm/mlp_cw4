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

    def pred(self, x, iid=None):
        return self(x).round()

    def from_data(acts, labels, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        acts, labels = acts.to(device), labels.to(device)
        probe = LRProbe(acts.shape[-1]).to(device)
        
        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        with open('accuracy_per_epoch.txt', 'w') as f:
            for epoch in tqdm(range(epochs), desc='Training LRProbe'):
                opt.zero_grad()
                outputs = probe(acts)
                loss = t.nn.BCELoss()(outputs, labels)
                loss.backward()
                opt.step()

                # Calculate accuracy
                preds = outputs.round()
                correct = (preds == labels).float().sum()
                accuracy = correct / labels.shape[0]
                
                # Write accuracy to file
                f.write(f'Epoch {epoch+1}: {accuracy.item()}\n')
        
        return probe

    @property
    def direction(self):
        return self.net[0].weight.data[0]

class MMProbeWrapper:
    def train_probe(self, acts, labels):
        self.model = MMProbe.from_data(acts, labels)

    def predict(self, acts):
        with t.no_grad():
            return self.model(acts).numpy()

class LRProbeWrapper:
    def train_probe(self, acts, labels):
        self.model = LRProbe.from_data(acts, labels)

    def predict(self, acts):
        with t.no_grad():
            return self.model(acts).numpy()


class MMProbe(t.nn.Module):
    def __init__(self, direction, covariance=None, inv=None, atol=1e-3):
        super().__init__()
        self.direction = t.nn.Parameter(direction, requires_grad=False)
        # if covariance is not None:
        #     # Ensure covariance is float32
        #     covariance = covariance.to(dtype=t.float32)
        #     # Compute the pseudo-inverse ensuring the operation stays in float32
        #     print(covariance.dtype)
        #     self.inv = t.nn.Parameter(t.linalg.pinv(covariance, hermitian=True, atol=atol).to(dtype=t.float32), requires_grad=False)
        # elif inv is not None:
        #     self.inv = t.nn.Parameter(inv, requires_grad=False)
        self.inv = t.nn.Parameter(inv, requires_grad=False)


    def forward(self, x, iid=False):
        if iid:
            return t.nn.Sigmoid()(x @ self.inv @ self.direction)
        else:
            #print("It's looking promising")
            return t.nn.Sigmoid()(x @ self.direction)


    def pred(self, x, iid=False):
        return self(x, iid=iid).round()

    def from_data(acts, labels, atol=1e-3, device='cpu'):
        acts, labels = acts.to(device).to(dtype=t.float32), labels.to(device).to(dtype=t.float32)
        pos_acts, neg_acts = acts[labels==1], acts[labels==0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean

        centered_data = t.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)

        # Ensure the covariance matrix is in float32 before passing it to MMProbe
        covariance = (centered_data.t() @ centered_data / acts.shape[0]).to(dtype=t.float32)
        covariance=None
    
        probe = MMProbe(direction, covariance=covariance).to(device)

        return probe