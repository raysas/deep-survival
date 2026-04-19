'''
pytest unit tests for SurvivalDataset class
'''
from src.dataset import SurvivalDataset
from sksurv.datasets import load_whas500  

def test_load_survival_dataset():
    data_x, data_y = load_whas500()
    dataset = SurvivalDataset(data_x, data_y)
    assert len(dataset) == len(data_x) == len(data_y), "dataset length should match input data length"

def test_getitem():
    data_x, data_y = load_whas500()
    dataset = SurvivalDataset(data_x, data_y)
    x, event, time = dataset[0]
    assert x.shape[0] == data_x.shape[1], "feature shape should match input data shape"
    assert event in [0, 1], "event should be 0 or 1"
    assert isinstance(time, (int, float)), "time should be a number"

