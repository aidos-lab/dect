"""
Tests the ect functions.
"""

import torch
from torch.cuda import is_available

from dect.directions import generate_2d_directions, generate_uniform_directions
from dect.ect import compute_ect, compute_ect_point_cloud, compute_ect_points


def test_true():
    assert True


"""
Test the ECT for point clouds. 
"""


def test_compute_ect_case_points():
    """
    Test the `compute_ect` function for point clouds.
    """
    device = "cpu"
    # 2D Case
    seed = 2024
    num_points = 10
    g = torch.Generator(device=device).manual_seed(seed)
    v = generate_2d_directions(num_thetas=17).to(device)
    x = torch.rand(size=(num_points, 2))
    ect = compute_ect(x, v=v, radius=1, resolution=13, scale=10)

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
    seed = 2024
    num_points = 10
    g = torch.Generator(device=device).manual_seed(seed)
    v = generate_2d_directions(num_thetas=17).to(device)
    x = torch.rand(size=(num_points, 2), device=device)
    ect = compute_ect(x, v=v, radius=1, resolution=13, scale=10)

    assert ect.get_device() == 0  # 0 indicates cuda.

    # Check that min and max are 0 and num_pts
    torch.testing.assert_close(
        ect.max(), torch.tensor(num_points, dtype=torch.float32, device=device)
    )
    torch.testing.assert_close(
        ect.min(), torch.tensor(0.0, dtype=torch.float32, device=device)
    )


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


#     def test_ecc_multi_set_directions(self):
#         """
#         Check that the dimensions are correct.
#         lin of size [bump_steps, 1, 1, 1]
#         """
#         lin = torch.linspace(-1, 1, 13).view(-1, 1, 1, 1)
#         index = torch.tensor([0, 0, 0], dtype=torch.long)
#         nh = torch.tensor([[0.0, 0.0], [0.5, 0.5], [0.5, 0.5]])
#         scale = 100
#         ecc = compute_ecc(nh, index, lin, scale)
#         assert ecc.shape == (1, 1, 13, 2)

#     def test_ecc_multi_batch(self):
#         """
#         Check that the dimensions are correct.
#         lin of size [bump_steps, 1, 1, 1]
#         """
#         lin = torch.linspace(-1, 1, 13).view(-1, 1, 1, 1)
#         index = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
#         nh = torch.tensor([[0.0], [0.5], [0.5], [0.7], [0.7]])
#         scale = 100
#         ecc = compute_ecc(nh, index, lin, scale)
#         assert ecc.shape == (2, 1, 13, 1)

#         # Check that min and max are 0 and 1
#         torch.testing.assert_close(ecc[0].max(), torch.tensor(2.0))
#         torch.testing.assert_close(ecc[0].min(), torch.tensor(0.0))

#         torch.testing.assert_close(ecc[1].max(), torch.tensor(3.0))
#         torch.testing.assert_close(ecc[1].min(), torch.tensor(0.0))

#     def test_ecc_normalized(self):
#         """
#         Check that the dimensions are correct.
#         lin of size [bump_steps, 1, 1, 1]
#         """
#         lin = torch.linspace(-1, 1, 13).view(-1, 1, 1, 1)
#         index = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
#         nh = torch.tensor([[0.0], [0.5], [0.5], [0.7], [0.7]])
#         scale = 100
#         ecc = compute_ecc(nh, index, lin, scale)
#         assert ecc.shape == (2, 1, 13, 1)
#         ecc_normalized = normalize(ecc)

#         # Check that min and max are 0 and 1
#         torch.testing.assert_close(ecc_normalized[0].max(), torch.tensor(1.0))
#         torch.testing.assert_close(ecc_normalized[0].min(), torch.tensor(0.0))

#         torch.testing.assert_close(ecc_normalized[1].max(), torch.tensor(1.0))
#         torch.testing.assert_close(ecc_normalized[1].min(), torch.tensor(0.0))
