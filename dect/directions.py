"""
Functions to initialize directions in 2, 3 and $n$ dimensions.
"""

import itertools
import torch


def generate_uniform_directions(num_thetas: int, d: int, seed: int, device: str):
    """
    Generate randomly sampled directions from a sphere in d dimensions.

    A standard normal is sampled and projected onto the unit sphere to 
    yield a randomly sampled set of points on the unit spere. Please 
    note that the generated tensor has shape [d, num_thetas].

    Parameters
    ----------
    num_thetas: int
        The number of directions to generate.
    d: int
        The dimension of the unit sphere. Default is 3 (hence R^3)
    """
    g = torch.Generator(device=device).manual_seed(seed)
    v = torch.randn(size=(d, num_thetas), device=device, generator=g)
    v /= v.pow(2).sum(axis=0).sqrt()
    return v


def generate_2d_directions(num_thetas: int = 64):
    """
    Provides a structured set of directions in two dimensions. First the
    interval [0,2*pi] is devided into a regular grid and the corresponding
    angles on the unit circle calculated.

    Parameters
    ----------
    num_thetas: int
        The number of directions to generate.

    Returns
    ----------
    v: Tensor
        Tensor of shape [2,num_thetas] containing the directions where each
        column is one direction in 2D.
        The directions start at $theta=0$ and runs to $theta = 2 * pi$.
    """

    v = torch.vstack(
        [
            torch.sin(torch.linspace(0, 2 * torch.pi, num_thetas)),
            torch.cos(torch.linspace(0, 2 * torch.pi, num_thetas)),
        ]
    )

    return v


def generate_multiview_directions(num_thetas: int, d: int):
    """
    NOTE: Partially depreciated.

    Generates multiple sets of structured directions in n dimensions.

    We generate sets of directions by embedding the 2d unit circle in d
    dimensions and sample this unit circle in a structured fashion. This
    generates d choose 2 structured directions that are organized in channels,
    compatible with the ECT calculations.

    After computing the ECT, we obtain an d choose 2 channel image where each
    channel consists of a structured ect along a hyperplane. For the 3-d case we
    would obtain a 3 channel ect with direction sampled along the xy, xz and yz
    planes in three dimensions.

    Parameters
    ----------
    num_thetas: int
        The number of directions to generate.
    d: int
        The dimension of the unit sphere. Default is 3 (hence R^3)
    """
    w = torch.vstack(
        [
            torch.sin(torch.linspace(0, 2 * torch.pi, num_thetas)),
            torch.cos(torch.linspace(0, 2 * torch.pi, num_thetas)),
        ]
    )

    # We obtain n choose 2 channels.
    idx_pairs = list(itertools.combinations(range(d), r=2))

    v = torch.zeros(size=(len(idx_pairs), d, num_thetas))

    for idx, idx_pair in enumerate(idx_pairs):
        v[idx, idx_pair[0], :] = w[0]
        v[idx, idx_pair[1], :] = w[1]

    return v
