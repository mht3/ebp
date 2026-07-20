import torch.nn as nn

from ebp.tasks import CoordinateRegression, DatasetConfig
from ebp.optimizers import DerivativeFreeConfig, DerivativeFreeOptimizer

def test_derivative_free_optimizer():
    '''Tests DFO (similar to cross entropy optimizer)
    '''
    dataset = CoordinateRegression(DatasetConfig(dataset_size=10))
    bounds = dataset.get_target_bounds()

    config = DerivativeFreeConfig(bounds=bounds, train_samples=256)
    so = DerivativeFreeOptimizer.initialize(config, "cuda")

    negatives = so.sample(64, nn.Identity())
    assert negatives.shape == (64, config.train_samples, bounds.shape[1])