import torch
from ebp import models

def test_conv_net():
    '''
    Tests if output shape matches what we expect for passing a batch of two (3, 96, 96) cxhxw images.

    Architecture:
        Shape after being passed through:
            x: (2, 3, 96,96) input batch
            CoordConv: (2, 5, 96, 96)
            CNN (padding=1): (2, 32, 96, 96)
            1x1 Conv+Relu: (2, 16, 96, 96)
            Global Avg Pool: (2, 16)
            MLP: (2, 2)
    '''
    output_dim = 2
    hidden_depth = 2
    config = models.ConvMLPConfig(
        cnn_config=models.CNNConfig(5),
        mlp_config=models.MLPConfig(16, 128, output_dim, hidden_depth),
        spatial_reduction=models.SpatialReduction.AVERAGE_POOL,
        coord_conv=True,
    )

    net = models.ConvMLP(config)

    # input is two random rgb images
    x = torch.randn(2, 3, 96, 96)
    with torch.no_grad():
        # predict a 2D (x, y) coordinate for coordinate regression task
        out = net(x)
    
    assert out.shape == (x.shape[0], output_dim)