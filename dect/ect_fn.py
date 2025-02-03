"""Elementwise operations for the calculation of the ECC"""

from typing import TypeAlias
import torch

Tensor: TypeAlias = torch.Tensor
"""@private"""


def indicator(ecc: Tensor)  -> Tensor:
    """
    Indicator function for the calculation of the ect.

    Args:
        ecc (Tensor):
            The height values.

    Returns:
        Tensor: The ECC.
    """
    return torch.heaviside(ecc, torch.tensor([0.0]))


def scaled_sigmoid(ecc: Tensor) -> Tensor:
    """
    Sigmoid function for the calculation of the ect.

    Args:
        ecc (Tensor):
        The height values.

    Returns:
        Tensor: The ECC.
    """
    return torch.sigmoid(ecc)
