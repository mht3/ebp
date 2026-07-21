import numpy as np
import torch
from ebp.tasks import CoordinateRegression, DatasetConfig

def test_coordinate_regression():
    config = DatasetConfig(dataset_size=30, seed=0)
    dataset = CoordinateRegression(config)

    assert len(dataset) == 30

    image, target = dataset[0]

    assert image.shape == (3, 96, 96)
    assert image.dtype == torch.float32
    assert image.min() >= 0.0 and image.max() <= 1.0

    assert target.shape == (2,)
    assert target.dtype == torch.float32
    assert (target >= -1.0).all() and (target <= 1.0).all()

    assert dataset.coordinates_scaled.shape == (30, 2)

    assert (dataset.get_target_bounds() == np.array([[-1.0, -1.0], [1.0, 1.0]])).all()
