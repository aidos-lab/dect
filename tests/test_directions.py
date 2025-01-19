import torch
from dect.directions import generate_uniform_directions


def test_generate_uniform_directions_shape():
    d = 3
    num_thetas = 13
    v = generate_uniform_directions(num_thetas=num_thetas, d=d, seed=10, device="cpu")
    assert v.shape == (d, num_thetas)

    d = 30
    num_thetas = 129
    v = generate_uniform_directions(num_thetas=num_thetas, d=d, seed=10, device="cpu")
    assert v.shape == (d, num_thetas)


def test_generate_uniform_directions_seed_correct():
    d = 3
    num_thetas = 13
    v1 = generate_uniform_directions(num_thetas=num_thetas, d=d, seed=10, device="cpu")
    v2 = generate_uniform_directions(num_thetas=num_thetas, d=d, seed=10, device="cpu")
    assert torch.equal(v1, v2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = 6
    num_thetas = 17
    v1 = generate_uniform_directions(num_thetas=num_thetas, d=d, seed=10, device=device)
    v2 = generate_uniform_directions(num_thetas=num_thetas, d=d, seed=10, device=device)
    assert torch.equal(v1, v2)


def test_generate_uniform_directions_norm_correct():
    d = 3
    num_thetas = 13
    v1 = generate_uniform_directions(num_thetas=num_thetas, d=d, seed=10, device="cpu")
    v2 = generate_uniform_directions(num_thetas=num_thetas, d=d, seed=10, device="cpu")
    assert torch.equal(v1, v2)

    device = "cpu"
    d = 6
    num_thetas = 17
    v1 = generate_uniform_directions(num_thetas=num_thetas, d=d, seed=10, device=device)

    assert torch.allclose(
        v1.norm(dim=0), torch.ones(size=(num_thetas,), dtype=torch.float32)
    )
