"""Elementwise operations for the calculation of the ECC"""

import torch
from torch import Tensor


def indicator(ecc: Tensor) -> Tensor:
    """
    Indicator function for the calculation of the ect.

    Args:
        ecc:
            The height values.

    Returns:
        Tensor: The ECC.
    """
    return torch.heaviside(ecc, torch.tensor([0.0], device=ecc.device))


def scaled_sigmoid(ecc: Tensor) -> Tensor:
    """
    Sigmoid function for the calculation of the ect.

    Args:
        ecc: The height values.

    Returns:
        The ECC.
    """
    return torch.sigmoid(ecc)
