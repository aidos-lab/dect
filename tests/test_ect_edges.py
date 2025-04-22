"""
Tests the core functions for computing the
ECT over a edges.
"""

import pytest
import torch
import warnings

from dect.directions import generate_uniform_directions
from dect.ect import compute_ect_edges


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_compute_ect_edges_noindex(device):
    """
    Test the `compute_ect` function for point clouds.
    """

    # Check if device is available, else skip the test.
    if not getattr(torch, device).is_available():
        warnings.warn(f"Device {device} not available, skipping the tests.")
        return

    seed = 2024
    ambient_dimension = 5
    num_points = 10
    v = generate_uniform_directions(
        num_thetas=17, seed=seed, device=device, d=ambient_dimension
    ).to(device)
    x = torch.rand(size=(num_points, ambient_dimension), device=device)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]], device=device)
    ect = compute_ect_edges(
        x, edge_index=edge_index, v=v, radius=1, resolution=13, scale=10
    )

    assert ect.device.type == device

    # TODO: Implement proper tests that the ect has been computed correctly.
    # Most likely a parametrized set of fixtures are the way to go here.
