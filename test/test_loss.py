'''
testing main loss functionalities of a torch loss while still compatible for surv data
'''

import torch
from src.loss import NegativeLogPartialLikelihood

def test_negative_log_partial_likelihood():
    loss_fn = NegativeLogPartialLikelihood()

    risk_scores = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32)
    event = torch.tensor([1, 0, 1], dtype=torch.float32)  # Event occurred for samples 1 and 3
    time = torch.tensor([5.0, 10.0, 7.0], dtype=torch.float32)

    target = (event, time)

    loss = loss_fn(risk_scores, target)
    print('neg log partial likelihood loss:', loss.item())

def test_nn_functionality():
    loss_fn = NegativeLogPartialLikelihood()

    risk_scores = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32, requires_grad=True)
    event = torch.tensor([1, 0, 1], dtype=torch.float32)  # Event occurred for samples 1 and 3
    time = torch.tensor([5.0, 10.0, 7.0], dtype=torch.float32)

    target = (event, time)

    loss = loss_fn(risk_scores, target)
    print('loss before backward:', loss.item())

    loss.backward()
    print('gradients:', risk_scores.grad)