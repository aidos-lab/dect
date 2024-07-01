"""
Helper function to generate a structured set of directions in 2 and 3
dimensions.
"""

import itertools
import torch


def generate_uniform_directions(
    num_thetas: int = 64, d: int = 3, device: str = "cpu"
):
    """
    Generate randomly sampled directions from a sphere in d dimensions.

    First a standard gaussian centered at 0 with standard deviation 1 is sampled
    and then projected onto the unit sphere. This yields a uniformly sampled set
    of points on the unit spere. Please note that the generated shapes with have
    shape [d, num_thetas].

    Parameters
    ----------
    num_thetas: int
        The number of directions to generate.
    d: int
        The dimension of the unit sphere. Default is 3 (hence R^3)
    device: str
        The device to put the tensor on.
    """
    v = torch.randn(size=(d, num_thetas), device=device)
    v /= v.pow(2).sum(axis=0).sqrt().unsqueeze(1)
    return v


def generate_uniform_2d_directions(num_thetas: int = 64, device: str = "cpu"):
    """
    Generate uniformly sampled directions on the unit circle in two dimensions.

    Provides a structured set of directions in two dimensions. First the
    interval [0,2*pi] is devided into a regular grid and the corresponding
    angles on the unit circle calculated.

    Parameters
    ----------
    num_thetas: int
        The number of directions to generate.
    d: int
        The dimension of the unit sphere. Default is 3 (hence R^3)
    device: str
        The device to put the tensor on.
    """
    v = torch.vstack(
        [
            torch.sin(
                torch.linspace(0, 2 * torch.pi, num_thetas, device=device)
            ),
            torch.cos(
                torch.linspace(0, 2 * torch.pi, num_thetas, device=device)
            ),
        ]
    )

    return v


def generate_multiview_directions(num_thetas: int, bump_steps: int, d: int):
    """
    Generates multiple sets of structured directions in n dimensions.

    We generate sets of directions by embedding the 2d unit circle in d
    dimensions and sample this unit circle in a structured fashion. This
    generates d choose 2 structured directions that are organized in channels,
    compatible with the ECT calculations.

    After computing the ECT, we obtain an d choose 2 channel image where each
    channel consists of a structured ect along a hyperplane. For the 3-d case we
    would obtain a 3 channel ect with direction sampled along the xy, xz and yz
    planes in three dimensions.
    """
    w = torch.vstack(
        [
            torch.sin(torch.linspace(0, 2 * torch.pi, bump_steps)),
            torch.cos(torch.linspace(0, 2 * torch.pi, bump_steps)),
        ]
    )

    # We obtain n choose 2 channels.
    idx_pairs = list(itertools.combinations(range(d), r=2))

    v = torch.zeros(size=(len(idx_pairs), d, num_thetas))

    for idx, idx_pair in enumerate(idx_pairs):
        v[idx, idx_pair[0], :] = w[0]
        v[idx, idx_pair[1], :] = w[1]

    return v
