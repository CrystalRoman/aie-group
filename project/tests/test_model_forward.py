import torch

from src.models.factory import create_model


def test_unet_forward_shape():
    model = create_model(
        "unet",
        in_channels=1,
        out_channels=1,
        base_channels=8,
        encoder_name="swin_tiny_patch4_window7_224",
        pretrained=False,
        img_size=256,
    )
    model.eval()

    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, 1, 256, 256)
