"""
Helper function to generate a structured set of directions in 2 and 3 dimensions.
"""

import torch


def generate_uniform_directions(num_thetas: int = 64, d: int = 3, device: str = "cpu"):
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

    Provides a structured set of directions in two dimensions. First the interval
    [0,2*pi] is devided into a regular grid and the corresponding angles on the
    unit circle calculated.

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
            torch.sin(torch.linspace(0, 2 * torch.pi, num_thetas, device=device)),
            torch.cos(torch.linspace(0, 2 * torch.pi, num_thetas, device=device)),
        ]
    )

    return v


# Needs some work.
def generate_structured_directions(
    num_thetas: int = 64, d: int = 3, device: str = "cpu"
):
    """
    Generates a structured set of directions in either 2 or 3 dimensions.
    If the dimension is two, we evenly space the directions along the unit
    circle.
    In the 3d case we sample along three unit circles, the xy, xz and yz planes.
    Along these unit circles we sample evenly. A check is performed to ensure
    the number of directions provided is divisible by three.
    """
    assert d == 3 or d == 2, "You should provide either 2 or 3 for the dimension."

    if d == 3:
        if num_thetas % 3 != 0:
            raise ValueError(
                f"Number of theta's should be divisible by three, you provided {num_thetas}"
            )
        w1 = torch.vstack(
            [
                torch.sin(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
                torch.cos(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
                torch.zeros_like(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
            ]
        )

        w2 = torch.vstack(
            [
                torch.sin(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
                torch.zeros_like(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
                torch.cos(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
            ]
        )

        w3 = torch.vstack(
            [
                torch.zeros_like(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
                torch.sin(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
                torch.cos(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
            ]
        )
        v = torch.hstack([w1, w2, w3])
    elif d == 2:
        v = torch.vstack(
            [
                torch.sin(torch.linspace(0, 2 * torch.pi, num_thetas, device=device)),
                torch.cos(torch.linspace(0, 2 * torch.pi, num_thetas, device=device)),
            ]
        )
    else:
        raise ValueError("d should be either 2 or three")
    return v
