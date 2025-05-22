"""
Tests the core functions for computing the
ECT over a point cloud.
"""

import pytest
import torch

from dect.directions import generate_uniform_directions
from dect.ect import compute_ect_points

"""
Test the ECT for 
"""


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_compute_ect_case_points_noindex_cpu(device):
    """
    Test the `compute_ect` function for point clouds.
    """

    # Check if device is available, else skip the test.
    if not getattr(torch, device).is_available():
        return

    # 2D Case
    seed = 2024
    ambient_dimension = 5
    num_points = 10
    v = generate_uniform_directions(
        num_thetas=17, seed=seed, device=device, d=ambient_dimension
    ).to(device)
    x = torch.rand(size=(num_points, ambient_dimension), device=device)
    ect = compute_ect_points(x, v=v, radius=1, resolution=13, scale=10)

    assert ect.device.type == device

    # Check that min and max are 0 and num_pts
    torch.testing.assert_close(
        ect.max(), torch.tensor(num_points, dtype=torch.float32, device=device)
    )
    torch.testing.assert_close(
        ect.min(), torch.tensor(0.0, dtype=torch.float32, device=device)
    )
