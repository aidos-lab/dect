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

    # We obtain n choose 2 channels.
    idx_pairs = list(itertools.combinations(range(d), r=2))

    num_directions_per_circle = num_thetas // len(idx_pairs)
    remainder = num_thetas % len(idx_pairs)

    w = torch.vstack(
        [
            torch.sin(
                torch.linspace(0, 2 * torch.pi, num_directions_per_circle + remainder)
            ),
            torch.cos(
                torch.linspace(0, 2 * torch.pi, num_directions_per_circle + remainder)
            ),
        ]
    )

    multiview_dirs = []
    for idx, idx_pair in enumerate(idx_pairs):
        num_t = num_directions_per_circle
        if idx == 0 and remainder != 0:
            num_t = num_directions_per_circle + remainder

        v = torch.zeros(size=(d, num_t))
        v[idx_pair[0], :] = w[0, :num_t]
        v[idx_pair[1], :] = w[1, :num_t]

        multiview_dirs.append(v)

    return torch.hstack(multiview_dirs)


def generate_spherical_grid_directions(num_thetas: int, num_phis: int, d: int = 3):
    """
    Generates a smooth spherical grid of directions on the unit sphere in 3D using
    latitude–longitude (θ, φ) style sampling.

    The directions are parameterized by θ (polar angle, [0, π]) and φ (azimuthal angle, [0, 2π)),
    and returned as a tensor of shape [3, num_thetas * num_phis], with each column a unit vector.

    Parameters
    ----------
    num_thetas: int
        Number of θ samples (from 0 to π, inclusive).
    num_phis: int
        Number of φ samples (from 0 to 2π, exclusive).
    d: int
        Must be 3, as spherical coordinates are for 3D.

    Returns
    -------
    Tensor of shape [3, num_thetas * num_phis] containing unit vectors on the sphere.
    """
    assert d == 3, "Spherical coordinates are only defined for d=3."
    theta = torch.linspace(0, torch.pi, num_thetas)
    phi = torch.linspace(0, 2 * torch.pi, num_phis, endpoint=False)
    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing='ij')  # shape [num_phis, num_thetas]
    sin_theta = torch.sin(theta_grid)
    x = sin_theta * torch.cos(phi_grid)
    y = sin_theta * torch.sin(phi_grid)
    z = torch.cos(theta_grid)
    dirs = torch.stack([x, y, z], dim=0)  # [3, num_phis, num_thetas]
    dirs = dirs.reshape(3, -1)  # [3, num_thetas*num_phis]
    return dirs
