'''
main user interface for training and evaluating the DeepSurv model
'''

import torch
import torch.nn as nn
import numpy as np

from src.dataset import SurvivalDataset
from src.loss import NegativeLogPartialLikelihood
from sksurv.datasets import load_whas500

class VanillaDeepSurv(nn.Module):
    def __init__(self, input_dim, hidden_layers=[100], dropout=0.5):
        super().__init__()
        layers = []
        for i, hidden_dim in enumerate(hidden_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_layers[i-1], hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)
        last_dim = hidden_layers[-1] if hidden_layers else input_dim
        self.risk_head = nn.Linear(last_dim, 1)

    def forward(self, x):
        hidden = self.network(x)
        return self.risk_head(hidden).squeeze(-1) #to get it at the end as a 1dim tesnor


def _prepare_features(data_x):
    # WHAS500 includes categorical columns => one-hot encoding
    encoded = data_x.copy()
    if hasattr(encoded, "select_dtypes"):
        cat_cols = encoded.select_dtypes(include=["category", "object"]).columns
        if len(cat_cols) > 0:
            encoded = encoded.drop(columns=cat_cols)
    return encoded.astype("float32")



from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

def risk_groups(model, dataset):
    model.eval()
    with torch.no_grad():
        full_x = torch.from_numpy(np.asarray(dataset.X, dtype=np.float32))
        risk_scores = model(full_x).cpu().numpy()
    median_risk = np.median(risk_scores)
    risk_group = np.where(risk_scores >= median_risk, 'high', 'low')
    
    # -- KM plot
    kmf = KaplanMeierFitter()
    for group in ['high', 'low']:
        mask = risk_group == group
        kmf.fit(dataset.time[mask], event_observed=dataset.event[mask], label=group)
        kmf.plot_survival_function()
    plt.title('Kaplan-Meier risk groups')
    plt.show()

if __name__ == "__main__":
    data_x, data_y = load_whas500()
    features = _prepare_features(data_x)

    dataset = SurvivalDataset(features, data_y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = VanillaDeepSurv(input_dim=features.shape[1], hidden_layers=[100], dropout=0.5)
    loss_fn = NegativeLogPartialLikelihood()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(50):
        epoch_loss = 0.0
        for batch_x, batch_event, batch_time in dataloader:
            batch_x = batch_x.float()
            batch_event = batch_event.float()
            batch_time = batch_time.float()

            risk_scores = model(batch_x)
            loss = loss_fn(risk_scores, (batch_event, batch_time))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(dataloader)
        print(f"-- epoch {epoch + 1}/50 | training loss: {mean_loss:.4f}")

    risk_groups(model, dataset)