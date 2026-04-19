'''
Survival loss module:

- Partial likelihood: negative log partial likelihood for Cox model

This is a child of nn.Module and can be used as a loss function in training the DeepSurv model
'''

import torch
import torch.nn as nn

class NegativeLogPartialLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogPartialLikelihood, self).__init__()

    def forward(self, risk_scores,target):
        '''
        compute the negative log partial likelihood for Cox model

        parameters:
        -----------
            risk_scores: tensor, predicted risk scores from the model
            target: tuple, (event, time) survival data
                event: tensor, binary indicator of event occurrence (delta)
                time: tensor, observed time (T)

        returns:
        --------
            loss: tensor, negative log partial likelihood loss

        formula:
            L = -sum_{i: delta_i=1} (r_i - log(sum_{j: T_j >= T_i} exp(r_j)))

        [!NOTE] to calculate the loss it's important to define a risk set at each iteration  
        [!NOTE] no ties assumption (breslow method)
        '''
        event, time = target

        sorted_indices = torch.argsort(time, descending=True)
        risk_scores = risk_scores[sorted_indices]
        event = event[sorted_indices]
        time = time[sorted_indices]

        cumulative_risk = torch.cumsum(torch.exp(risk_scores), dim=0)
        log_cumulative_risk = torch.log(cumulative_risk)    
        loss = -torch.sum(event * (risk_scores - log_cumulative_risk))

        return loss