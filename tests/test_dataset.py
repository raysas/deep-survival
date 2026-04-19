from src.dataset import SurvivalDataset

from sksurv.datasets import load_whas500


def test_survival_dataset_loads_whas500():
	data_x, data_y = load_whas500()
	dataset = SurvivalDataset(data_x, data_y)
	assert len(dataset) == len(data_x)


def test_survival_dataset_getitem_returns_triplet():
	data_x, data_y = load_whas500()
	dataset = SurvivalDataset(data_x, data_y)
	x, event, time = dataset[0]
	assert x.shape[0] == data_x.shape[1]
	assert event in [0, 1, True, False]
	assert isinstance(time, (int, float))