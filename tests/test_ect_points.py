"""
Tests the core functions for computing the
ECT over a point cloud.
"""

import torch

from dect.directions import generate_2d_directions, generate_uniform_directions
from dect.ect import compute_ect_points

"""
Test the ECT for 
"""


def test_compute_ect_case_points():
    """
    Test the `compute_ect` function for point clouds.
    """
    device = "cpu"
    # 2D Case
    seed = 2024
    ambient_dimension = 5
    num_points = 10
    v = generate_uniform_directions(
        num_thetas=17, seed=seed, device=device, d=ambient_dimension
    ).to(device)
    x = torch.rand(size=(num_points, ambient_dimension))
    ect = compute_ect_points(x, v=v, radius=1, resolution=13, scale=10)

    assert ect.get_device() == -1

    # Check that min and max are 0 and num_pts
    torch.testing.assert_close(ect.max(), torch.tensor(num_points, dtype=torch.float32))
    torch.testing.assert_close(ect.min(), torch.tensor(0.0, dtype=torch.float32))


def test_compute_ect_case_points_cuda():
    """
    Test the `compute_ect` function for point clouds on the gpu.
    """
    if not torch.cuda.is_available():
        return

    device = "cuda:0"
    # 2D Case
    num_points = 10
    v = generate_2d_directions(num_thetas=17).to(device)
    x = torch.rand(size=(num_points, 2), device=device)
    ect = compute_ect_points(x, v=v, radius=1, resolution=13, scale=10)

    assert ect.get_device() == 0  # 0 indicates cuda.

    # Check that min and max are 0 and num_pts
    torch.testing.assert_close(
        ect.max(), torch.tensor(num_points, dtype=torch.float32, device=device)
    )
    torch.testing.assert_close(
        ect.min(), torch.tensor(0.0, dtype=torch.float32, device=device)
    )
