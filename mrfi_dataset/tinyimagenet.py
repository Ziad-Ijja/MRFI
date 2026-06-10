import json
import os
from pathlib import Path
from urllib.request import urlopen

import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

tf = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize,
])


def _get_wnid_to_imagenet_idx():
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    imagenet_class_index = json.load(urlopen(url))
    return {v[0]: int(k) for k, v in imagenet_class_index.items()}


class ImageFolderWithImageNetTargets(data.Dataset):
    """
    ImageFolder dataset that remaps targets to their original ImageNet indices (0-999),
    so a pretrained ResNet/VGG/etc. can be evaluated directly with Acc_experiment.

    The dataset also exposes `valid_mask` (BoolTensor of shape [1000]) which is True
    for the 200 ImageNet classes present in Tiny ImageNet.  Use `register_mask_hook`
    to attach it to the model so MRFI's Acc_experiment works out of the box.
    """

    def __init__(self, root, transform=None):
        self.base = datasets.ImageFolder(
            str(Path(root).expanduser()), transform=transform
        )

        wnid_to_imagenet_idx = _get_wnid_to_imagenet_idx()

        # tiny_idx (0-199)  ->  imagenet_idx (0-999)
        self.target_map = torch.tensor(
            [wnid_to_imagenet_idx[wnid] for wnid in self.base.classes]
        )

        # Boolean mask: True for the 200 valid ImageNet indices
        self.valid_mask = torch.zeros(1000, dtype=torch.bool)
        self.valid_mask[self.target_map] = True

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, tiny_target = self.base[idx]
        imagenet_target = self.target_map[tiny_target].item()
        return image, imagenet_target


# ── module that masks the 800 absent classes ──────────────────────────────────

class _MaskInvalidClasses(nn.Module):
    """Sets logits of the 800 absent ImageNet classes to -inf."""

    def __init__(self, valid_mask: torch.BoolTensor):
        super().__init__()
        # Register as a buffer so it moves with .cuda() / .to()
        self.register_buffer("invalid_mask", ~valid_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, self.invalid_mask] = float("-inf")
        return x


def register_mask_hook(model: nn.Module, valid_mask: torch.BoolTensor) -> None:
    """
    Attach a forward hook to `model` that zeroes out the 800 logits
    corresponding to ImageNet classes absent from Tiny ImageNet.


    Parameters
    ----------
    model      : the nn.Module (or MRFI wrapper) to patch
    valid_mask : BoolTensor[1000] — True for the 200 valid ImageNet indices
                 (returned by `get_testset().valid_mask`)
    """
    masker = _MaskInvalidClasses(valid_mask)

    # If model is on CUDA, put the masker there too
    try:
        device = next(model.parameters()).device
        masker = masker.to(device)
    except StopIteration:
        pass

    def _hook(module, input, output):
        return masker(output)

    return model.register_forward_hook(_hook)


# ── singleton testset & public helpers ────────────────────────────────────────

_testset: ImageFolderWithImageNetTargets | None = None


def get_testset(
    folder: str = "~/dataset/val/tiny-imagenet-200/val",
) -> ImageFolderWithImageNetTargets:
    global _testset
    if _testset is None:
        _testset = ImageFolderWithImageNetTargets(
            os.path.expanduser(folder), transform=tf
        )
    return _testset


def make_testloader(
    size: int | None = None,
    folder: str = "~/dataset/val/tiny-imagenet-200/val",
    **kwargs,
) -> data.DataLoader:
    testset = get_testset(folder)
    if size is None:
        return data.DataLoader(testset, **kwargs)
    subset = data.Subset(testset, range(size))
    return data.DataLoader(subset, **kwargs)