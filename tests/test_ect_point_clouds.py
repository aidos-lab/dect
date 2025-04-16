"""
Tests the core functions for computing the
ECT over a point cloud. A point cloud is of
shape [B,N,D] where the first dimension is the
batch, the second is the number of points and the
third is the ambient dimension.
Note that in this format each point in the point
cloud has to have the same cardinality.
"""

import torch

from dect.directions import generate_2d_directions, generate_uniform_directions
from dect.ect import compute_ect_point_cloud


def test_compute_ect_point_cloud_case_cpu():
    """
    Tests the ECT computation of a point cloud.
    Differs in that it expects an input shape of
    size [B,N,D], where B is the batch size,
    N is the number of points and D is the ambient
    dimension.
    """
    ambient_dimension = 4
    num_pts = 100
    batch_size = 8
    num_thetas = 100
    seed = 0
    x = torch.rand(size=(batch_size, num_pts, ambient_dimension))
    v = generate_uniform_directions(
        num_thetas=num_thetas, d=ambient_dimension, device="cpu", seed=seed
    )

    ect = compute_ect_point_cloud(x, v, radius=10, resolution=30, scale=500)

    assert ect[0].max() == num_pts
    assert ect[0].min() == 0


def test_compute_ect_point_cloud_case_cuda():
    """
    Tests the ECT computation of a point cloud.
    Differs in that it expects an input shape of
    size [B,N,D], where B is the batch size,
    N is the number of points and D is the ambient
    dimension.
    """
    ambient_dimension = 4
    num_pts = 100
    batch_size = 8
    num_thetas = 100
    seed = 0
    x = torch.rand(size=(batch_size, num_pts, ambient_dimension), device="cuda")
    v = generate_uniform_directions(
        num_thetas=num_thetas, d=ambient_dimension, device="cuda", seed=seed
    )

    ect = compute_ect_point_cloud(x, v, radius=10, resolution=30, scale=500)

    assert ect[0].max() == num_pts
    assert ect[0].min() == 0
