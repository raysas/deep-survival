'''
dataset module: house of SurvivalDataset pytorch Dataset object
'''

import numpy as np
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
        # Convert common tabular inputs (e.g. pandas DataFrame) to an array for stable row indexing.
        self.X = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        try:
            self.X = self.X.astype(np.float32)
        except (TypeError, ValueError):
            # Keep original dtype when conversion is not possible; callers can preprocess features.
            pass
        self.event, self.time = self._parse_target(target)

        assert len(self.X) == len(self.event) == len(self.time), "Length of X, event, and time must be the same"

    def _parse_target(self, target):
        # sksurv datasets commonly return a structured array with two fields: event and time.
        if hasattr(target, "dtype") and getattr(target.dtype, "names", None):
            field_names = target.dtype.names
            if len(field_names) < 2:
                raise ValueError("Structured target must contain at least event and time fields")
            event = np.asarray(target[field_names[0]])
            time = np.asarray(target[field_names[1]])
            return event, time

        if isinstance(target, (tuple, list)) and len(target) == 2:
            first = np.asarray(target[0])
            second = np.asarray(target[1])

            # Accept either (event, time) or (time, event).
            if np.issubdtype(first.dtype, np.bool_) or np.array_equal(np.unique(first), [0, 1]) or np.array_equal(np.unique(first), [0]) or np.array_equal(np.unique(first), [1]):
                return first, second
            if np.issubdtype(second.dtype, np.bool_) or np.array_equal(np.unique(second), [0, 1]) or np.array_equal(np.unique(second), [0]) or np.array_equal(np.unique(second), [1]):
                return second, first
            return first, second

        raise TypeError("target must be a structured array or a tuple/list of (event, time) or (time, event)")

    def __len__(self):
        return len(self.X)

    def _to_python_scalar_if_possible(self, value):
        arr = np.asarray(value)
        return arr.item() if arr.shape == () else arr
    
    def __getitem__(self, idx):
        x = np.asarray(self.X[idx], dtype=np.float32)
        event = self._to_python_scalar_if_possible(self.event[idx]) # to fix the issue of getting a 0-dim array instead of a scalar when indexing (used in KM plot)
        time = self._to_python_scalar_if_possible(self.time[idx])
        return x, event, time
    
