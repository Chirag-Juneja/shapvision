import numpy as np
import torch
from torchvision import transforms


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def preprocess(img, custom_transforms=None):
    if custom_transforms:
        transform = custom_transforms
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return transform(img).unsqueeze(0)


def postprocess(x):
    inv_transform = [
        transforms.Lambda(nhwc_to_nchw),
        transforms.Normalize(
            mean=(-1 * np.array(mean) / np.array(std)).tolist(),
            std=(1 / np.array(std)).tolist(),
        ),
        transforms.Lambda(nchw_to_nhwc),
    ]
    inv_transform = transforms.Compose(inv_transform)
    return inv_transform(x)


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x
