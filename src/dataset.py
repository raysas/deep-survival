'''
dataset module: house of SurvivalDataset pytorch Dataset object
'''

from torch.utils.data import Dataset

class SurvivalDataset(Dataset):
    '''
    SurvivalDataset:
        pytorch Dataset that is built to be compatible with survival data format
        extensions can be made as child of this dataset depending on application or model used

    '''
    def __init__(self, X, target):
        '''
        SurvivalDataset constructor

        parameters:
        -----------
            X: df or array-like, covariate data (features)
            target: tuple, survival data (time, event) 
        '''
        self.X = X
        self.event= target[0]
        self.time = target[1]

        assert len(self.X) == len(self.event) == len(self.time), "Length of X, event, and time must be the same"

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.event[idx], self.time[idx]